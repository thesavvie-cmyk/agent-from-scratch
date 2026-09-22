"""OpenTelemetry instrumentation bootstrap (block 23).

All agentkit modules that emit spans import ``get_tracer`` and
``add_span_event`` from here.  When OTel is not installed (or when
``setup_tracing`` has not been called / ``exporter_type="none"``), both
functions return zero-cost no-ops — no guards needed at call sites.

Environment variables
---------------------
AGENTKIT_OTEL_FILE:
    JSONL output path for the file exporter.
    Defaults to ``results/traces/trace.jsonl``.

AGENTKIT_CAPTURE_CONTENT:
    ``"false"`` (default) | ``"true"``
    When ``"true"``, LLM prompt and completion text are stored in span
    attributes (useful for debugging; may be large).

Usage
-----
    # Entry point — configure once:
    from agentkit.telemetry import setup_tracing
    setup_tracing(service_name="my-agent", exporter_type="file")

    # Inside any agentkit module (no guards):
    from agentkit.telemetry import get_tracer
    tracer = get_tracer()
    with tracer.start_as_current_span("my.op") as span:
        span.set_attribute("key", "value")
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Self

# ── OTel availability ─────────────────────────────────────────────────────────

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,  # re-exported so tests need only one import
    )

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    InMemorySpanExporter = None  # type: ignore[assignment,misc]

# ── Environment ───────────────────────────────────────────────────────────────

CAPTURE_CONTENT: bool = os.getenv("AGENTKIT_CAPTURE_CONTENT", "false").lower() == "true"
_DEFAULT_TRACE_FILE = os.getenv("AGENTKIT_OTEL_FILE", "results/traces/trace.jsonl")


# ── No-op implementations ─────────────────────────────────────────────────────


class _NoOpSpan:
    """Context-manager span stub — every method is a no-op."""

    __slots__ = ()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def set_attribute(self, key: str, value: object) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def record_exception(self, exc: BaseException, attributes: dict[str, Any] | None = None) -> None:
        pass


_NOOP_SPAN = _NoOpSpan()


class _NoOpTracer:
    """Tracer stub — returns ``_NOOP_SPAN`` for every operation."""

    __slots__ = ()

    def start_as_current_span(self, name: str, **_kwargs: Any) -> _NoOpSpan:
        return _NOOP_SPAN


# ── Module-level singleton (replaced by setup_tracing) ────────────────────────

_tracer: Any = _NoOpTracer()


def get_tracer() -> Any:
    """Return the active tracer.  Zero-cost when OTel is not configured."""
    return _tracer


def add_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """Add an event to the current active OTel span.

    No-op when OTel is not installed or no span is active.
    """
    if not _OTEL_AVAILABLE:
        return
    span = _otel_trace.get_current_span()
    if span.is_recording():
        span.add_event(name, attributes=attributes or {})


def infer_llm_system(model: str) -> str:
    """Derive ``gen_ai.system`` from a LiteLLM model string."""
    m = model.lower()
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gpt" in m or "openai" in m:
        return "openai"
    if "gemini" in m or "google" in m:
        return "google_ai_studio"
    if "mistral" in m:
        return "mistral_ai"
    return "unknown"


# ── File exporter ─────────────────────────────────────────────────────────────

if _OTEL_AVAILABLE:

    class FileSpanExporter(SpanExporter):
        """Append-only JSONL exporter — one span per line."""

        def __init__(self, filepath: str | Path = _DEFAULT_TRACE_FILE) -> None:
            self._path = Path(filepath)

        def export(self, spans: Any) -> SpanExportResult:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                for span in spans:
                    fh.write(json.dumps(_span_to_dict(span)) + "\n")
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            pass

else:
    FileSpanExporter = None  # type: ignore[assignment,misc]


def _span_to_dict(span: Any) -> dict[str, Any]:
    """Convert a ReadableSpan to a JSON-serialisable dict."""
    ctx = span.context
    parent_id = (
        format(span.parent.span_id, "016x")
        if span.parent and span.parent.span_id
        else None
    )
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": parent_id,
        "name": span.name,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "duration_ms": (
            round((span.end_time - span.start_time) / 1_000_000, 2)
            if span.end_time and span.start_time
            else None
        ),
        "attributes": dict(span.attributes) if span.attributes else {},
        "events": [
            {
                "name": e.name,
                "timestamp": e.timestamp,
                "attributes": dict(e.attributes) if e.attributes else {},
            }
            for e in (span.events or [])
        ],
        "status": span.status.status_code.name if hasattr(span.status, "status_code") else "UNSET",
    }


# ── Setup / teardown ──────────────────────────────────────────────────────────


def setup_tracing(
    service_name: str = "agentkit",
    exporter_type: str = "file",
    filepath: str | None = None,
    _provider: Any = None,
) -> Any:
    """Configure global tracing.

    Parameters
    ----------
    service_name:
        OTel ``service.name`` resource attribute.
    exporter_type:
        ``"file"``    — append spans as JSONL to *filepath*.
        ``"console"`` — print spans to stdout (debugging).
        ``"none"``    — keep no-op tracer; nothing is recorded.
    filepath:
        Output path for ``exporter_type="file"``.  Defaults to
        ``AGENTKIT_OTEL_FILE`` env var or ``results/traces/trace.jsonl``.
    _provider:
        Inject a pre-built ``TracerProvider``.  Used in tests to supply an
        ``InMemorySpanExporter`` without touching the real OTel global state.

    Returns
    -------
    The ``TracerProvider`` that was configured, or ``None`` for no-op.
    """
    global _tracer

    if exporter_type == "none":
        _tracer = _NoOpTracer()
        return None

    if _provider is not None:
        # Test/demo injection: only update our module-level tracer.
        # Do NOT call set_tracer_provider — it can only be set once globally
        # and triggers "Overriding not allowed" warnings on subsequent calls.
        _tracer = _provider.get_tracer(service_name)
        return _provider

    if not _OTEL_AVAILABLE:
        _tracer = _NoOpTracer()
        return None

    resource = Resource.create({"service.name": service_name})
    provider = _TracerProvider(resource=resource)

    if exporter_type == "file":
        exporter = FileSpanExporter(filepath or _DEFAULT_TRACE_FILE)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    elif exporter_type == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        raise ValueError(f"Unknown exporter_type: {exporter_type!r}")

    _otel_trace.set_tracer_provider(provider)
    _tracer = provider.get_tracer(service_name)
    return provider


def reset_tracing() -> None:
    """Reset to no-op tracer.  Call in test teardown to avoid state leakage."""
    global _tracer
    _tracer = _NoOpTracer()
