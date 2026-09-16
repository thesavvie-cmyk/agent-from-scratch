"""Document analysis tools: Excel, PDF, images, audio stub (block 11).

PDF and image analysis use the Anthropic SDK directly because they require
multimodal message blocks (document/image) that LiteLLM wraps inconsistently.

Audio note
----------
Anthropic has no audio transcription model, so transcribe_audio is a stub
that returns an explicit error with alternatives. Adding faster-whisper would
cost ~500 MB and runs at ~1x realtime on 2 CPU cores — too heavy for the
current server profile. The interface is preserved so callers can swap in
their own implementation.
"""
from __future__ import annotations

import base64
import os
from typing import Any

from agentkit.config import FAST_MODEL
from agentkit.context import ExecutionContext
from agentkit.tools.base import tool
from agentkit.tools.files import WorkspaceEscapeError, _get_workspace

_PDF_MAX_BYTES = 32 * 1024 * 1024   # 32 MB (Anthropic limit)
_IMG_MAX_BYTES = 5 * 1024 * 1024    # 5 MB
_EXCEL_MAX_ROWS = 200
_EXCEL_MAX_COLS = 20

_IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _sdk_model() -> str:
    return FAST_MODEL.removeprefix("anthropic/")


def _anthropic_client() -> Any:
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


# ── Excel ──────────────────────────────────────────────────────────────────────


@tool
def read_excel(
    context: ExecutionContext,
    path: str,
    sheet: str | None = None,
) -> str:
    """Read an Excel (.xlsx) file and return it as a Markdown table.

    path: file path relative to workspace root
    sheet: sheet name (default: first/active sheet)

    Large sheets are truncated at 200 rows × 20 columns with a size note.
    """
    try:
        import openpyxl  # type: ignore[import]
    except ImportError:
        return "Error: openpyxl is not installed. Run: uv add openpyxl"

    ws = _get_workspace(context)
    try:
        p = ws.resolve(path)
    except WorkspaceEscapeError as exc:
        return f"Error: {exc}"
    if not p.exists():
        return f"Error: file not found: {path!r}"

    try:
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001
        return f"Error loading workbook: {exc}"

    if sheet is not None:
        if sheet not in wb.sheetnames:
            avail = ", ".join(wb.sheetnames)
            return f"Error: sheet {sheet!r} not found. Available: {avail}"
        active = wb[sheet]
    else:
        active = wb.active

    rows: list[list[str]] = []
    for i, row in enumerate(active.iter_rows(values_only=True)):  # type: ignore[union-attr]
        if i >= _EXCEL_MAX_ROWS:
            break
        rows.append([str(c) if c is not None else "" for c in list(row)[:_EXCEL_MAX_COLS]])

    if not rows:
        return "(empty sheet)"

    total_rows = active.max_row or "?"
    total_cols = active.max_column or "?"

    header, *data_rows = rows
    sep = ["---"] * len(header)
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in data_rows:
        md_lines.append("| " + " | ".join(r) + " |")

    result = "\n".join(md_lines)
    shown_r, shown_c = len(rows), len(rows[0]) if rows else 0
    if shown_r >= _EXCEL_MAX_ROWS or shown_c >= _EXCEL_MAX_COLS:
        result += (
            f"\n\n[Showing first {shown_r} of {total_rows} rows, "
            f"{shown_c} of {total_cols} columns]"
        )
    return result


# ── PDF ────────────────────────────────────────────────────────────────────────


@tool
def analyze_pdf(context: ExecutionContext, path: str, query: str) -> str:
    """Analyze a PDF by sending it to Claude as a document block.

    path: PDF file path relative to workspace root
    query: question to answer about the document

    The entire PDF (up to 32 MB) is base64-encoded and sent to the
    Anthropic API in a single messages.create call.
    """
    ws = _get_workspace(context)
    try:
        p = ws.resolve(path)
    except WorkspaceEscapeError as exc:
        return f"Error: {exc}"
    if not p.exists():
        return f"Error: file not found: {path!r}"
    if p.suffix.lower() != ".pdf":
        return f"Error: {path!r} is not a PDF (got {p.suffix!r})"

    size = p.stat().st_size
    if size > _PDF_MAX_BYTES:
        mb = size / 1024 / 1024
        return f"Error: PDF is {mb:.1f} MB; limit is {_PDF_MAX_BYTES // 1024 // 1024} MB"

    try:
        client = _anthropic_client()
    except RuntimeError as exc:
        return f"Error: {exc}"

    data = base64.standard_b64encode(p.read_bytes()).decode()
    try:
        response = client.messages.create(
            model=_sdk_model(),
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": data,
                            },
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ],
        )
        return response.content[0].text  # type: ignore[index]
    except Exception as exc:  # noqa: BLE001
        return f"Error analyzing PDF: {exc}"


# ── Image ──────────────────────────────────────────────────────────────────────


@tool
def analyze_image(context: ExecutionContext, path: str, query: str) -> str:
    """Analyze an image using Claude's vision capabilities.

    path: image file path relative to workspace root (JPEG/PNG/GIF/WebP)
    query: question or instruction about the image

    File size limit: 5 MB. The image is base64-encoded and sent to the
    Anthropic API in a single messages.create call.
    """
    ws = _get_workspace(context)
    try:
        p = ws.resolve(path)
    except WorkspaceEscapeError as exc:
        return f"Error: {exc}"
    if not p.exists():
        return f"Error: file not found: {path!r}"

    media_type = _IMAGE_MEDIA_TYPES.get(p.suffix.lower())
    if media_type is None:
        supported = ", ".join(_IMAGE_MEDIA_TYPES)
        return f"Error: unsupported format {p.suffix!r}. Supported: {supported}"

    size = p.stat().st_size
    if size > _IMG_MAX_BYTES:
        mb = size / 1024 / 1024
        return f"Error: image is {mb:.1f} MB; limit is {_IMG_MAX_BYTES // 1024 // 1024} MB"

    try:
        client = _anthropic_client()
    except RuntimeError as exc:
        return f"Error: {exc}"

    data = base64.standard_b64encode(p.read_bytes()).decode()
    try:
        response = client.messages.create(
            model=_sdk_model(),
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ],
        )
        return response.content[0].text  # type: ignore[index]
    except Exception as exc:  # noqa: BLE001
        return f"Error analyzing image: {exc}"


# ── Audio (stub) ───────────────────────────────────────────────────────────────


@tool
def transcribe_audio(context: ExecutionContext, path: str) -> str:
    """[NOT IMPLEMENTED] Audio transcription is not supported.

    path: audio file path (ignored)

    Anthropic has no audio transcription model. Alternatives:
      - OpenAI Whisper API  (requires OPENAI_API_KEY, ~$0.006/min)
      - faster-whisper      (local ONNX, ~500 MB download, ~1x–2x realtime on CPU)
      - AssemblyAI / Deepgram (dedicated STT APIs with their own keys)

    To add transcription: implement a custom @tool using one of the above
    and register it with your Agent alongside the other document tools.
    """
    return (
        "Audio transcription is not implemented. "
        "Anthropic has no audio model. "
        "Use the OpenAI Whisper API, faster-whisper (local), or AssemblyAI."
    )
