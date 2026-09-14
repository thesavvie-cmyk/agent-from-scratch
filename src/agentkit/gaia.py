from __future__ import annotations

import asyncio
import time
from typing import Any

import litellm
from pydantic import BaseModel, ValidationError

from agentkit.config import require_env
from agentkit.prompts import GAIA_SYSTEM_PROMPT

# Per-provider concurrency limits (acquire before each API call)
PROVIDER_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
_PROVIDER_LIMITS: dict[str, int] = {"anthropic": 5}


class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""


# ── internal helpers ──────────────────────────────────────────────────────────

def _get_semaphore(model: str) -> asyncio.Semaphore:
    provider = model.split("/")[0] if "/" in model else "default"
    if provider not in PROVIDER_SEMAPHORES:
        PROVIDER_SEMAPHORES[provider] = asyncio.Semaphore(
            _PROVIDER_LIMITS.get(provider, 5)
        )
    return PROVIDER_SEMAPHORES[provider]


async def _raw_solve(
    model: str, question: str
) -> tuple[GaiaOutput, int, int, bool]:
    """Call the model and return (output, input_tokens, output_tokens, api_refusal)."""
    semaphore = _get_semaphore(model)
    async with semaphore:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": GAIA_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format=GaiaOutput,
            num_retries=2,
        )

    finish_reason: str | None = response.choices[0].finish_reason
    content: str | None = response.choices[0].message.content
    input_tokens: int = getattr(response.usage, "prompt_tokens", 0) or 0
    output_tokens: int = getattr(response.usage, "completion_tokens", 0) or 0
    api_refusal: bool = finish_reason == "refusal"

    if api_refusal or content is None:
        return (
            GaiaOutput(is_solvable=False, unsolvable_reason="Model refused to answer"),
            input_tokens,
            output_tokens,
            api_refusal,
        )

    try:
        output = GaiaOutput.model_validate_json(content)
    except ValidationError as exc:
        output = GaiaOutput(
            is_solvable=False,
            unsolvable_reason=f"Response parse error: {str(exc)[:120]}",
        )

    return output, input_tokens, output_tokens, api_refusal


# ── public API ────────────────────────────────────────────────────────────────

_WEB_KW = ("search", "browser", "web")
_BLOCK_KW = (
    "video", "pdf", "image", "audio", "excel", "word",
    "spreadsheet", "powerpoint", "docx", "mp3", "mp4", "youtube",
)


def _get_tools_str(problem: dict[str, Any]) -> str:
    """Extract the Tools string from Annotator Metadata."""
    import json as _json

    meta = problem.get("Annotator Metadata", {})
    if isinstance(meta, str):
        try:
            meta = _json.loads(meta)
        except (ValueError, TypeError):
            return ""
    return meta.get("Tools", "") if isinstance(meta, dict) else ""


def load_search_tasks(limit: int | None = None) -> list[dict[str, Any]]:
    """Level-1 tasks that require web search only (no file attachment, no
    video/PDF/image parsing).

    These are the tasks where a web-search agent can plausibly outperform a
    tool-less baseline, and they are the target evaluation set for chapter 8.
    """
    problems = load_level1()
    result = [
        p
        for p in problems
        if (
            any(kw in _get_tools_str(p).lower() for kw in _WEB_KW)
            and not p.get("file_name")
            and not any(kw in _get_tools_str(p).lower() for kw in _BLOCK_KW)
        )
    ]
    if limit is not None:
        result = result[:limit]
    return result


def load_file_tasks(limit: int | None = None) -> list[dict[str, Any]]:
    """Level-1 tasks that have a file attachment (target set for chapter 5+)."""
    problems = load_level1()
    result = [p for p in problems if p.get("file_name")]
    if limit is not None:
        result = result[:limit]
    return result


def load_level1(limit: int | None = None) -> list[dict[str, Any]]:
    """Load GAIA level-1 validation split. Caches locally via datasets library.

    Raises MissingEnvError if HF_TOKEN is not set.
    Raises RuntimeError if the dataset cannot be loaded (no access, etc.).
    """
    token = require_env("HF_TOKEN")
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_level1",
            split="validation",
            token=token,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load GAIA dataset. "
            "Ensure HF_TOKEN is valid and you have accepted the dataset terms at "
            "https://huggingface.co/datasets/gaia-benchmark/GAIA\n"
            f"Original error: {exc}"
        ) from exc

    problems: list[dict[str, Any]] = [dict(row) for row in ds]
    if limit is not None:
        problems = problems[:limit]
    return problems


async def solve_problem(model: str, question: str) -> GaiaOutput:
    """Call the model on a GAIA question. Returns a GaiaOutput."""
    output, _, _, _ = await _raw_solve(model, question)
    return output


def is_correct(prediction: str | None, answer: str) -> bool:
    """Exact match after strip and lowercase."""
    if prediction is None:
        return False
    return prediction.strip().lower() == answer.strip().lower()


async def evaluate_single(problem: dict[str, Any], model: str) -> dict[str, Any]:
    """Run one problem/model pair. Exceptions are captured into 'error', not raised."""
    task_id: str = str(problem.get("task_id", ""))
    question: str = str(problem.get("Question", ""))
    answer: str = str(problem.get("Final answer", ""))

    start = time.perf_counter()
    error: str | None = None
    output: GaiaOutput | None = None
    input_tokens = 0
    output_tokens = 0
    api_refusal = False

    try:
        output, input_tokens, output_tokens, api_refusal = await _raw_solve(model, question)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    latency_s = time.perf_counter() - start
    prediction: str | None = output.final_answer if (output and output.is_solvable) else None

    return {
        "task_id": task_id,
        "model": model,
        "correct": is_correct(prediction, answer),
        "is_solvable": output.is_solvable if output else False,
        "api_refusal": api_refusal,
        "prediction": prediction,
        "answer": answer,
        "unsolvable_reason": output.unsolvable_reason if output else "",
        "error": error,
        "latency_s": round(latency_s, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


async def run_experiment(
    problems: list[dict[str, Any]],
    models: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Run all problem/model pairs concurrently with a progress bar.

    Returns results grouped by model name.
    """
    from tqdm.asyncio import tqdm_asyncio

    tasks = [
        evaluate_single(problem, model)
        for problem in problems
        for model in models
    ]
    all_results: list[dict[str, Any]] = await tqdm_asyncio.gather(
        *tasks, desc="Evaluating"
    )

    by_model: dict[str, list[dict[str, Any]]] = {m: [] for m in models}
    for result in all_results:
        by_model[result["model"]].append(result)

    return by_model
