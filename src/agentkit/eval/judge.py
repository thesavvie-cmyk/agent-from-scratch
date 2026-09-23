"""LLM-as-judge — single verdict and pairwise comparison (block 24).

Design decisions
----------------
1. Reasoning before verdict — the prompt instructs the judge to write its
   analysis FIRST, then emit "VERDICT: PASS" or "VERDICT: FAIL" on a
   separate line.  This prevents the common failure mode where the model
   decides the verdict first and then constructs a post-hoc justification.

2. Trace summary included — trajectory rubrics (trajectory_soundness,
   data_provenance) cannot be evaluated from the final answer alone.
   The judge always receives TraceFeatures.summary_text() so it can check
   whether tool calls actually happened.

3. Pairwise order randomisation — judge_pairwise shuffles which response is
   shown first on every call.  The returned PairwiseVerdict records which
   was shown first so the caller can detect positional bias by running both
   orderings and comparing.

4. Separate judge model — the judge should NOT be the same model that is
   being evaluated.  Use SMART_MODEL (Sonnet) as judge, FAST_MODEL (Haiku)
   as agent.  This is enforced by caller convention, not by this module.

5. UNCLEAR verdict — if the judge output cannot be parsed (no "VERDICT:" line,
   or the response is truncated), the verdict is UNCLEAR rather than a random
   PASS/FAIL.  Callers should treat UNCLEAR as a missing data point.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agentkit.eval.dataset import EvalCase
    from agentkit.eval.rubrics import Rubric
    from agentkit.eval.traces import TraceFeatures
    from agentkit.llm import LlmClient

# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class Verdict:
    """Result of judging one (case, output, rubric) triple."""

    rubric_name: str
    verdict: Literal["PASS", "FAIL", "UNCLEAR"]
    reasoning: str
    raw: str   # full model output, for debugging


@dataclass
class PairwiseVerdict:
    """Result of a pairwise comparison under one rubric.

    ``winner`` is in terms of the original A/B assignment given by the caller.
    ``first_shown`` records which was presented first to the judge so that
    positional bias can be measured across multiple runs.
    """

    rubric_name: str
    winner: Literal["A", "B", "TIE", "UNCLEAR"]
    first_shown: Literal["A", "B"]
    reasoning: str


# ── Prompt builders ────────────────────────────────────────────────────────────

_SINGLE_TEMPLATE = """\
You are an expert evaluator assessing an AI agent's response.

=== RUBRIC: {rubric_name} ===
{rubric_description}

PASS criteria: {pass_criterion}
FAIL criteria: {fail_criterion}

=== EXAMPLES ===
{examples}

=== TASK ===
{task_input}

=== EXPECTED ANSWER (if known) ===
{expected}

=== AGENT RESPONSE ===
{output}

=== TRACE SUMMARY ===
{trace_summary}

=== YOUR EVALUATION ===
Analyse whether the agent's response satisfies the rubric criteria.
Write your reasoning first (2-5 sentences), considering the task, response, and trace summary.
Then on a new line write exactly one of:
VERDICT: PASS
VERDICT: FAIL

Do not write anything after the VERDICT line."""


_PAIRWISE_TEMPLATE = """\
You are an expert evaluator comparing two AI agent responses.

=== RUBRIC: {rubric_name} ===
{rubric_description}

PASS criteria: {pass_criterion}
FAIL criteria: {fail_criterion}

=== TASK ===
{task_input}

=== EXPECTED ANSWER (if known) ===
{expected}

=== RESPONSE {label_first} ===
{response_first}

=== RESPONSE {label_second} ===
{response_second}

=== YOUR EVALUATION ===
Compare both responses according to the rubric criteria.
Write your reasoning first (2-5 sentences).
Then on a new line write exactly one of:
WINNER: {label_first}
WINNER: {label_second}
WINNER: TIE

Do not write anything after the WINNER line."""


def _format_examples(rubric: Rubric) -> str:
    if not rubric.examples:
        return "(none)"
    parts = []
    for ex in rubric.examples[:2]:   # cap at 2 to keep prompt compact
        parts.append(
            f"Input: {ex.get('input', '')}\n"
            f"Output: {ex.get('output', '')}\n"
            f"Verdict: {ex.get('verdict', '')}\n"
            f"Reason: {ex.get('reason', '')}"
        )
    return "\n---\n".join(parts)


# ── Parsing helpers ────────────────────────────────────────────────────────────


def _parse_verdict(text: str) -> Literal["PASS", "FAIL", "UNCLEAR"]:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("VERDICT:"):
            token = line.split(":", 1)[1].strip().upper()
            if token in ("PASS", "FAIL"):
                return token  # type: ignore[return-value]
    return "UNCLEAR"


def _parse_winner(
    text: str,
    label_first: str,
    label_second: str,
) -> Literal["A", "B", "TIE", "UNCLEAR"]:
    """Parse WINNER: <label_first|label_second|TIE> and map back to A/B."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("WINNER:"):
            token = line.split(":", 1)[1].strip().upper()
            if "TIE" in token:
                return "TIE"
            if label_first.upper() in token:
                return "A" if label_first == "A" else "B"
            if label_second.upper() in token:
                return "A" if label_second == "A" else "B"
    return "UNCLEAR"


def _split_reasoning(text: str) -> str:
    """Return everything before the VERDICT/WINNER line."""
    lines = text.strip().splitlines()
    reasoning_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("VERDICT:", "WINNER:")):
            break
        reasoning_lines.append(line)
    return "\n".join(reasoning_lines).strip()


# ── Judge functions ────────────────────────────────────────────────────────────


async def judge_single(
    llm: LlmClient,
    case: EvalCase,
    output: str,
    trace_features: TraceFeatures | None,
    rubric: Rubric,
) -> Verdict:
    """Evaluate one (case, output) pair under *rubric*.

    Parameters
    ----------
    llm:
        The judge LLM (should be SMART_MODEL, not the same as the agent).
    case:
        The eval case (provides task input and expected answer).
    output:
        The agent's final answer string.
    trace_features:
        Extracted trace statistics.  Pass ``None`` for non-trajectory rubrics.
    rubric:
        The rubric to apply.
    """
    from agentkit.llm import LlmRequest
    from agentkit.types import Message

    trace_summary = (
        trace_features.summary_text() if trace_features is not None else "No trace available."
    )

    prompt = _SINGLE_TEMPLATE.format(
        rubric_name=rubric.name,
        rubric_description=rubric.description,
        pass_criterion=rubric.pass_criterion,
        fail_criterion=rubric.fail_criterion,
        examples=_format_examples(rubric),
        task_input=case.input or "(empty)",
        expected=case.expected or "Not specified",
        output=output or "(empty)",
        trace_summary=trace_summary,
    )

    request = LlmRequest(
        contents=[Message(role="user", content=prompt)],
    )
    response = await llm.generate(request)

    raw = ""
    if response.content:
        from agentkit.types import Message as Msg
        texts = [item.content for item in response.content if isinstance(item, Msg)]
        raw = texts[0] if texts else ""

    return Verdict(
        rubric_name=rubric.name,
        verdict=_parse_verdict(raw),
        reasoning=_split_reasoning(raw),
        raw=raw,
    )


async def judge_pairwise(
    llm: LlmClient,
    case: EvalCase,
    output_a: str,
    output_b: str,
    rubric: Rubric,
) -> PairwiseVerdict:
    """Compare two agent outputs for the same case under *rubric*.

    The order in which A and B are presented is randomised.  The caller
    should run this function in both orderings and compare to measure
    positional bias.
    """
    from agentkit.llm import LlmRequest
    from agentkit.types import Message

    # Randomise presentation order
    if random.random() < 0.5:
        first_shown: Literal["A", "B"] = "A"
        label_first, label_second = "A", "B"
        response_first, response_second = output_a, output_b
    else:
        first_shown = "B"
        label_first, label_second = "B", "A"
        response_first, response_second = output_b, output_a

    prompt = _PAIRWISE_TEMPLATE.format(
        rubric_name=rubric.name,
        rubric_description=rubric.description,
        pass_criterion=rubric.pass_criterion,
        fail_criterion=rubric.fail_criterion,
        task_input=case.input or "(empty)",
        expected=case.expected or "Not specified",
        label_first=label_first,
        label_second=label_second,
        response_first=response_first or "(empty)",
        response_second=response_second or "(empty)",
    )

    request = LlmRequest(
        contents=[Message(role="user", content=prompt)],
    )
    response = await llm.generate(request)

    raw = ""
    if response.content:
        from agentkit.types import Message as Msg
        texts = [item.content for item in response.content if isinstance(item, Msg)]
        raw = texts[0] if texts else ""

    winner = _parse_winner(raw, label_first, label_second)
    return PairwiseVerdict(
        rubric_name=rubric.name,
        winner=winner,
        first_shown=first_shown,
        reasoning=_split_reasoning(raw),
    )


# ── Convenience: cache key ─────────────────────────────────────────────────────


def verdict_cache_key(case_id: str, output: str, rubric_name: str) -> str:
    """Return a stable hash key for caching a verdict."""
    import hashlib

    text = f"{case_id}||{rubric_name}||{output}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def verdicts_to_dict(verdicts: list[Verdict]) -> dict[str, Any]:
    return {v.rubric_name: {"verdict": v.verdict, "reasoning": v.reasoning} for v in verdicts}
