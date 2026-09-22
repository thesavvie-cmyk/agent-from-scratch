# Tracing with OpenTelemetry

agentkit instruments all agent operations using [OpenTelemetry](https://opentelemetry.io/).
Tracing is **off by default** — zero overhead until you call `setup_tracing()`.

## Install

```bash
uv sync --group tracing
```

The `tracing` group adds `opentelemetry-sdk` and the OTLP exporter.
The rest of agentkit works fine without it.

## Quick start

```python
from agentkit.telemetry import setup_tracing

# Write spans to a JSONL file:
setup_tracing(service_name="my-agent", exporter_type="file")

# Or print spans to stdout (debugging):
setup_tracing(exporter_type="console")

# Then run your agent as normal — spans are emitted automatically.
```

## Span hierarchy

```
workflow.sequential.run  (or workflow.parallel.run)
└── agent.run
    └── agent.step
        ├── gen_ai.chat          (one per LLM call)
        └── tool.execute         (one per tool call)
            └── agent.run        (for AgentTool — child agent, same trace_id)
                └── agent.step
                    └── gen_ai.chat
```

Child agents spawned via `AgentTool` automatically inherit the parent trace
because `AgentTool.execute` is an `await` chain — OTel context propagates
through Python's `contextvars` without any extra code.

`asyncio.gather` in `ParallelWorkflow` copies the current `contextvars.Context`
to each task, so parallel agent spans are siblings under the same
`workflow.parallel.run` span.

## Span attributes

| Span | Key attributes |
|------|----------------|
| `gen_ai.chat` | `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` |
| `agent.run` | `agent.name`, `agent.max_steps`, `agent.steps_used` |
| `agent.step` | `agent.name`, `agent.step_num` |
| `tool.execute` | `tool.name`, `tool.status` |
| `workflow.*.run` | `workflow.num_steps` |

### Span events

When context compaction fires inside `_prepare_llm_request`, a
`context.compacted` event is added to the current `agent.step` span:

```
context.compacted  original_tokens=8192  final_tokens=2048  strategy=Truncate
```

### Capturing prompt/completion text

By default, LLM prompt and completion text are **not** stored in spans
(they can be large and sensitive).  Set the environment variable to enable:

```bash
AGENTKIT_CAPTURE_CONTENT=true uv run python my_agent.py
```

## View traces

### Terminal tree

```bash
uv run python scripts/trace_tree.py results/traces/trace.jsonl
# Filter to one trace:
uv run python scripts/trace_tree.py results/traces/trace.jsonl --trace abc123
```

### Jaeger (local Docker)

```bash
docker run -d --name jaeger \
  -p 16686:16686 -p 4317:4317 \
  jaegertracing/all-in-one:latest

# Then configure agentkit to export via OTLP:
from agentkit.telemetry import setup_tracing
setup_tracing(service_name="my-agent", exporter_type="otlp")
# (requires opentelemetry-exporter-otlp-proto-grpc)
```

Open <http://localhost:16686> and search for service `my-agent`.

### Arize Phoenix (local, no Docker)

```bash
pip install arize-phoenix
python -m phoenix.server.main &

from agentkit.telemetry import setup_tracing
setup_tracing(exporter_type="otlp")
```

Open <http://localhost:6006>.

## Run the demo

```bash
# Mocked demo (no API key needed):
uv run --group tracing python experiments/ch10_tracing.py --section all

# View the file exporter output:
uv run python scripts/trace_tree.py results/traces/trace.jsonl
```
