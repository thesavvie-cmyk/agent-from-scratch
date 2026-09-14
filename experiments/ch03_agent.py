"""Chapter 3 agent demo: web search, schema generation, simple_agent_loop.

Sections:
  A — search_web raw vs format_results
  B — generated schema vs handwritten schema
  C — Nobel Prize 2025 (live search)
  D — Kipchoge full task: search + calculator
  E — negative rule: Pythagorean theorem
"""
from __future__ import annotations

import json

from agentkit.config import SMART_MODEL
from agentkit.loop import simple_agent_loop
from agentkit.schema import function_to_tool_definition
from agentkit.tools.web import format_results, search_web
from agentkit.tools_manual import CALCULATOR_TOOL_DEFINITION, calculator


def sep(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


SYSTEM = "You are a helpful research assistant."


# ── A. Raw results vs formatted ───────────────────────────────────────────────
sep("A. search_web: raw output vs format_results")

results = search_web("Kipchoge marathon world record", max_results=3)
if isinstance(results, list):
    raw_chars = sum(len(str(r)) for r in results)
    keys = list(results[0].keys()) if results else []
    formatted = format_results(results)
    print(f"\nRaw: {len(results)} results, {raw_chars} chars, keys={keys}")
    print(f"Formatted: {len(formatted)} chars\n")
    print(formatted)
else:
    print(f"Error: {results}")


# ── B. Generated vs handwritten schema ───────────────────────────────────────
sep("B. Generated schema vs handwritten calculator schema")

generated = function_to_tool_definition(search_web)
print("\n[Generated — search_web]")
print(json.dumps(generated, indent=2))

print("\n[Handwritten — calculator (block 3)]")
print(json.dumps(CALCULATOR_TOOL_DEFINITION, indent=2))

print("\n[What the generated schema lacks vs handwritten:]")
print("  - No per-parameter descriptions — model only sees 'type: string'")
print("  - No enum for 'topic' — model may hallucinate invalid values")
print("  - str | None → 'string' — the None option is invisible to the model")
print("  - time_range valid values ('day'/'week'/'month'/'year') not documented")
print("  → Handwritten schemas are more expensive to maintain but safer for structured params")


# ── C. Nobel 2025 ─────────────────────────────────────────────────────────────
sep("C. simple_agent_loop — Nobel Prize in Physics 2025")

answer_c, history_c = simple_agent_loop(
    SYSTEM,
    "Who won the Nobel Prize in Physics in 2025?",
    [search_web],
    max_rounds=5,
)
tool_msgs_c = [m for m in history_c if m["role"] == "tool"]
print(f"\nSearch calls : {len(tool_msgs_c)}")
print(f"Answer       : {answer_c[:200]!r}")


# ── D. Kipchoge full task ─────────────────────────────────────────────────────
sep("D. Kipchoge task — search + calculator, SMART_MODEL, expected answer: 17")

KIPCHOGE_Q = (
    "If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, "
    "how many thousand hours would it take him to run the distance between the Earth "
    "and the Moon at its closest approach? "
    "Use the minimum perigee value on the Wikipedia page for the Moon. "
    "Round to the nearest 1000 hours."
)

answer_d, history_d = simple_agent_loop(
    SYSTEM, KIPCHOGE_Q, [search_web, calculator], model=SMART_MODEL, max_rounds=12
)

print(f"\nAnswer   : {answer_d!r}")
print("\nCall chain:")
for m in history_d:
    if m["role"] == "assistant" and m.get("tool_calls"):
        for tc in m["tool_calls"]:
            args_preview = tc["function"]["arguments"][:70].replace("\n", " ")
            print(f"  → {tc['function']['name']}({args_preview})")
    elif m["role"] == "tool":
        preview = m["content"][:100].replace("\n", " ")
        print(f"  ← {preview!r}")


# ── E. Negative rule — Pythagorean theorem ────────────────────────────────────
sep("E. Negative rule: 'what is the Pythagorean theorem'")

Q_PYTH = "What is the Pythagorean theorem?"

print("\n[Without negative rule]")
ans_no, hist_no = simple_agent_loop(SYSTEM, Q_PYTH, [search_web], max_rounds=4)
searched_no = any(m["role"] == "tool" for m in hist_no)
print(f"  Used search : {searched_no}")
print(f"  Answer      : {ans_no[:150]!r}")

SYSTEM_RESTRICTED = (
    SYSTEM
    + " Only use search_web for current events, breaking news, or information "
    "that may have changed after your training cutoff. For well-established "
    "facts, mathematical concepts, definitions, and historical events, "
    "answer directly from your knowledge without searching."
)

print("\n[With negative rule]")
ans_yes, hist_yes = simple_agent_loop(
    SYSTEM_RESTRICTED, Q_PYTH, [search_web], max_rounds=4
)
searched_yes = any(m["role"] == "tool" for m in hist_yes)
print(f"  Used search : {searched_yes}")
print(f"  Answer      : {ans_yes[:150]!r}")
