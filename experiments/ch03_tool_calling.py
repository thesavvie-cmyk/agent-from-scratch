"""Chapter 3 demo: manual tool calling, step by step.

Sections:
  A — tool_calls absent vs present
  B — manual steps 4-5 with litellm.set_verbose to show Anthropic wire format
  C — run_with_tools: two rounds
  D — bad schema vs good schema
  E — error resilience: division by zero
"""
from __future__ import annotations

import json

import litellm

from agentkit.config import FAST_MODEL
from agentkit.tools_manual import (
    CALCULATOR_TOOL_DEFINITION,
    execute_tool_call,
    run_with_tools,
)

litellm.suppress_debug_info = True


def sep(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


# ── A. tool_calls absent vs present ──────────────────────────────────────────
sep("A. tool_calls: absent (factual) vs present (calculation)")

for q in ["What is the capital of South Korea?", "What is 1234 * 5678?"]:
    r = litellm.completion(
        model=FAST_MODEL,
        messages=[{"role": "user", "content": q}],
        tools=[CALCULATOR_TOOL_DEFINITION],
    )
    m = r.choices[0].message
    tc_summary = None
    if m.tool_calls:
        tc = m.tool_calls[0]
        tc_summary = f"name={tc.function.name!r}  args={tc.function.arguments}"
    print(f"\nQ: {q!r}")
    print(f"   content    : {m.content!r}")
    print(f"   tool_calls : {tc_summary or '(none)'}")


# ── B. Manual loop, step by step ─────────────────────────────────────────────
sep("B. Manual steps — 1234 * 5678 (set_verbose shows Anthropic wire format)")

question = "What is 1234 * 5678?"
messages: list[dict] = [{"role": "user", "content": question}]

print("\n[Step 1] Messages sent to model:")
print(json.dumps(messages, indent=2))

# set_verbose=True prints the raw Anthropic API request/response to stdout,
# showing how LiteLLM converts OpenAI tools → Anthropic input_schema format
print("\n[Enabling litellm.set_verbose — raw Anthropic request/response follows]")
litellm.set_verbose = True
resp1 = litellm.completion(
    model=FAST_MODEL,
    messages=messages,
    tools=[CALCULATOR_TOOL_DEFINITION],
)
litellm.set_verbose = False
print("[set_verbose off]\n")

msg1 = resp1.choices[0].message
print("[Step 2] Assistant reply:")
print(f"  content    : {msg1.content!r}")
if msg1.tool_calls:
    tc = msg1.tool_calls[0]
    print(f"  tool_calls : id={tc.id!r}  name={tc.function.name!r}  args={tc.function.arguments}")

    result = execute_tool_call(tc)
    print(f"\n[Step 3] execute_tool_call → {result!r}")

    messages.append(
        {
            "role": "assistant",
            "content": msg1.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    print("\n[Step 4] Updated messages (tool result appended):")
    print(json.dumps(messages, indent=2))

    resp2 = litellm.completion(
        model=FAST_MODEL,
        messages=messages,
        tools=[CALCULATOR_TOOL_DEFINITION],
    )
    final = resp2.choices[0].message.content
    print(f"\n[Step 5] Final answer: {final!r}")


# ── C. run_with_tools — two rounds ───────────────────────────────────────────
sep("C. run_with_tools — (1234 * 5678) / 4321, two tool calls")

answer, history = run_with_tools(
    "Calculate (1234 * 5678) / 4321, rounded to the nearest integer."
)
tool_msgs = [m for m in history if m["role"] == "tool"]
print(f"\nAnswer  : {answer!r}")
print(f"Rounds  : {len(tool_msgs)} tool call(s)")
for i, tm in enumerate(tool_msgs, 1):
    print(f"  Call {i}: tool_call_id={tm['tool_call_id']!r}  result={tm['content']!r}")


# ── D. Bad schema vs good schema ─────────────────────────────────────────────
sep("D. Schema quality: bad vs good")

BAD_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": "does math",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "op": {"type": "string"},
            },
            "required": ["a", "b", "op"],
        },
    },
}

q = "What is 1234 * 5678?"
for label, tool_def in [("BAD ", BAD_TOOL), ("GOOD", CALCULATOR_TOOL_DEFINITION)]:
    r = litellm.completion(
        model=FAST_MODEL,
        messages=[{"role": "user", "content": q}],
        tools=[tool_def],
    )
    m = r.choices[0].message
    if m.tool_calls:
        tc = m.tool_calls[0]
        tc_str = f"name={tc.function.name!r}  args={tc.function.arguments}"
    else:
        tc_str = "(no tool call)"
    print(f"\n[{label}]")
    print(f"  content    : {m.content!r}")
    print(f"  tool_calls : {tc_str}")


# ── E. Error resilience: divide by zero ──────────────────────────────────────
sep("E. Error resilience: 'how much is 5 divided by 0'")

answer, history = run_with_tools("How much is 5 divided by 0?")
tool_msgs = [m for m in history if m["role"] == "tool"]
print(f"\nFinal answer: {answer!r}")
for i, tm in enumerate(tool_msgs, 1):
    print(f"  Tool result {i}: {tm['content']!r}")
