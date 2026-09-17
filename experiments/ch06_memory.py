"""Chapter 6 long-term memory experiments (block 14).

Sections
--------
a) Basic scenario: session 1 user shares info, session 2 new agent recalls it
b) Deduplication: same fact in 3 phrasings across 3 sessions, dedup table
c) Contradiction: "I live in Moscow" → "I moved to Saint Petersburg"
d) Noise: task session (marathon calc), check what got extracted
e) Isolation: two user_ids, verify no leakage

Usage
-----
    uv run python experiments/ch06_memory.py --section a
    uv run python experiments/ch06_memory.py --section all

Requires ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from agentkit.agent import Agent
from agentkit.config import FAST_MODEL
from agentkit.llm import LlmClient
from agentkit.memory.extraction import extract_memories
from agentkit.memory.longterm import LongTermMemory
from agentkit.memory.session import InMemorySessionStore

RESULTS_DIR = Path(__file__).parent.parent / "results"
MEMORY_DIR = RESULTS_DIR / "ch06_memory"


def _hr(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def _fresh_memory(subdir: str = "") -> LongTermMemory:
    """Create a fresh memory store for this experiment run."""
    from agentkit.embeddings import get_embedding_provider

    path = MEMORY_DIR / subdir if subdir else MEMORY_DIR
    path.mkdir(parents=True, exist_ok=True)
    # Clean previous data
    import shutil
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return LongTermMemory(
        path=str(path),
        embedding_provider=get_embedding_provider(),
    )


def _make_agent(mem: LongTermMemory, user_id: str, tools: list[Any] | None = None) -> Agent:
    return Agent(
        model=LlmClient(FAST_MODEL),
        tools=tools or [],
        instructions="You are a helpful assistant with long-term memory about users.",
        max_steps=4,
        memory=mem,
        user_id=user_id,
    )


# ── helpers to drain pending tasks ────────────────────────────────────────────

async def _flush_tasks() -> None:
    """Let fire-and-forget memory save tasks complete."""
    # give create_task() coroutines two event-loop iterations to finish
    for _ in range(5):
        await asyncio.sleep(0)


# ── Section A — cross-session recall ──────────────────────────────────────────


async def section_a() -> None:
    _hr("Section A — Cross-session recall")

    mem = _fresh_memory("a")
    user_id = "alice"
    store = InMemorySessionStore()

    # Session 1: user introduces herself
    agent1 = _make_agent(mem, user_id)
    session1 = store.create(user_id=user_id)
    print("\n  [Session 1] Alice introduces herself")
    inputs = [
        "Hi! My name is Alice. I'm a Python developer working on ML pipelines.",
        "I'm based in Amsterdam and I mainly use PyTorch and FastAPI.",
    ]
    for msg in inputs:
        print(f"  You: {msg}")
        r = await agent1.run(msg, session=session1)
        session1 = store.get(session1.session_id) or session1
        print(f"  Agent: {r.output}")
    await _flush_tasks()

    # Show what was extracted
    all_memories = mem.list_all(user_id)
    print(f"\n  Extracted memories ({len(all_memories)}):")
    for m in all_memories:
        print(f"    - {m.text}")

    # Session 2: NEW agent object, NEW session — only memory persists
    print("\n  [Session 2] New agent, new session — only long-term memory persists")
    agent2 = _make_agent(mem, user_id)
    session2 = store.create(user_id=user_id)
    question = "What framework do I use for ML?"
    print(f"  You: {question}")
    r2 = await agent2.run(question, session=session2)
    print(f"  Agent: {r2.output}")

    injected = session2.state.get("memory_context") or r2.context.state.get("memory_context", "")
    print(f"\n  Memory injected into context:\n  {injected}")

    ok = "pytorch" in str(r2.output).lower() or "torch" in str(r2.output).lower()
    print(f"\n  Result: {'PASS — recalled PyTorch from memory' if ok else 'FAIL (check output above)'}")


# ── Section B — deduplication ─────────────────────────────────────────────────


async def section_b() -> None:
    _hr("Section B — Deduplication across sessions")

    user_id = "bob"

    # Three phrasings of the same fact
    phrasings = [
        "My name is Bob.",
        "I'm Bob, by the way.",
        "People call me Bob.",
    ]

    print("\n  Adding three phrasings with different dedup thresholds:")
    print(f"\n  {'Threshold':>10}  {'Records stored':>15}  {'Scores'}")
    print(f"  {'─'*10}  {'─'*15}  {'─'*40}")

    for threshold in [0.95, 0.90, 0.80]:
        mem = _fresh_memory(f"b_{int(threshold*100)}")
        mem.dedup_threshold = threshold
        for phrase in phrasings:
            mem.add(user_id=user_id, text=phrase)

        all_mems = mem.list_all(user_id)
        # Search to get similarity scores between phrasings
        scores = []
        for phrase in phrasings[1:]:
            results = mem.search(user_id=user_id, query=phrase, top_k=1, min_score=0.0)
            if results:
                scores.append(f"{results[0].score:.3f}")

        score_str = ", ".join(scores)
        print(f"  {threshold:>10.2f}  {len(all_mems):>15}  {score_str}")

    # Detailed view at threshold=0.9
    print("\n  Detail at threshold=0.90:")
    mem90 = _fresh_memory("b_detail")
    mem90.dedup_threshold = 0.90
    for i, phrase in enumerate(phrasings):
        result = mem90.add(user_id=user_id, text=phrase)
        action = "created" if result.created_at == result.updated_at else "updated"
        print(f"    [{i+1}] '{phrase}' → {action} (id={result.id[:8]}...)")

    final = mem90.list_all(user_id)
    print(f"  Final records: {len(final)}")
    for m in final:
        print(f"    - \"{m.text}\"")


# ── Section C — contradiction ──────────────────────────────────────────────────


async def section_c() -> None:
    _hr("Section C — Contradiction: location update")

    mem = _fresh_memory("c")
    user_id = "carol"

    print("\n  Step 1: store initial location")
    m1 = mem.add(user_id=user_id, text="I live in Moscow.")
    print(f"    Added: \"{m1.text}\"  id={m1.id[:8]}  created={time.strftime('%H:%M:%S', time.localtime(m1.created_at))}")

    time.sleep(0.1)  # ensure updated_at differs

    print("\n  Step 2: store contradicting update")
    m2 = mem.add(user_id=user_id, text="I moved to Saint Petersburg last month.")
    print(f"    Added: \"{m2.text}\"  id={m2.id[:8]}  created={time.strftime('%H:%M:%S', time.localtime(m2.created_at))}")

    # What does search return?
    print("\n  search('where do I live?') BEFORE fix:")
    results_before = mem.search(user_id=user_id, query="where do I live?", top_k=5, min_score=0.0)
    for r in results_before:
        ts = time.strftime("%H:%M:%S", time.localtime(r.updated_at))
        print(f"    score={r.score:.3f}  updated={ts}  text=\"{r.text}\"")

    print("\n  Interpretation: search returns BOTH records.")
    print("  Resolution: results sorted by (score DESC, updated_at DESC)")
    print("  → newest fact appears first when scores are close.")

    # Confirm ordering
    if results_before:
        newest_first = results_before[0].updated_at >= results_before[-1].updated_at
        top_text = results_before[0].text
        print(f"\n  Top result: \"{top_text}\"")
        print(f"  Is newest first: {newest_first}")

    # Show what an agent would see: memory injected context
    agent = _make_agent(mem, user_id)
    r = await agent.run("Where do I currently live?")
    await _flush_tasks()
    injected = r.context.state.get("memory_context", "(none)")
    print(f"\n  Memory injected into context:\n  {injected}")
    print(f"\n  Agent answer: {r.output}")

    ok = "peter" in str(r.output).lower() or "петер" in str(r.output).lower() or "saint" in str(r.output).lower()
    print(f"\n  Result: {'PASS — agent saw the newer location' if ok else 'INFO — check output above'}")


# ── Section D — noise / task content ─────────────────────────────────────────


async def section_d() -> None:
    _hr("Section D — Noise: task session should NOT pollute memory")

    user_id = "dave"

    # Simulate events from a task-focused session (marathon calculator)
    from agentkit.types import Event, Message, ToolResult, ToolCall

    task_events = [
        Event(
            execution_id="x",
            author="agent",
            content=[Message(role="user", content=(
                "Calculate Kipchoge's average pace if he ran a marathon "
                "(42.195 km) in 2 hours 1 minute 9 seconds."
            ))],
        ),
        Event(
            execution_id="x",
            author="agent",
            content=[ToolCall(
                tool_call_id="c1", name="calculator",
                arguments={"expression": "42195 / (2*3600 + 69)"},
            )],
        ),
        Event(
            execution_id="x",
            author="agent",
            content=[ToolResult(
                tool_call_id="c1", name="calculator",
                status="success", content=["5.693 m/s"],
            )],
        ),
        Event(
            execution_id="x",
            author="agent",
            content=[Message(role="assistant", content=(
                "Kipchoge's average pace was approximately 5.693 m/s "
                "or 2:53 min/km."
            ))],
        ),
    ]

    print("\n  Transcript (task session):")
    for evt in task_events:
        for item in evt.content:
            if hasattr(item, "content") and hasattr(item, "role"):
                print(f"  [{item.role}]: {item.content[:100]}")
            elif hasattr(item, "name") and hasattr(item, "arguments"):
                print(f"  [tool_call]: {item.name}({item.arguments})")
            elif hasattr(item, "content") and hasattr(item, "status"):
                print(f"  [tool_result]: {item.content[0][:80]}")

    llm = LlmClient(FAST_MODEL)
    print("\n  Extracting memories (no existing)…")
    facts = await extract_memories(llm, task_events, existing=None)

    print(f"\n  Extracted facts ({len(facts)}):")
    if facts:
        for f in facts:
            print(f"    - {f}")
        marathon_leaked = any(
            any(kw in f.lower() for kw in ["kipchoge", "marathon", "5.693", "pace", "km", "min/km"])
            for f in facts
        )
        print(f"\n  Marathon facts leaked into memory: {marathon_leaked}")
        if marathon_leaked:
            print("  [BAD] Extraction prompt needs tightening — task content leaked")
        else:
            print("  [OK] Task content correctly excluded")
    else:
        print("  (none — correct: no stable user facts in this session)")
        print("  [OK] Extraction correctly returned empty for a task-only session")

    # Now add a mixed session where user mentions themselves
    mixed_events = task_events + [
        Event(
            execution_id="x",
            author="agent",
            content=[Message(role="user", content=(
                "Thanks! By the way, I'm training for my first marathon. "
                "I'm 35 years old and I live in Berlin."
            ))],
        ),
        Event(
            execution_id="x",
            author="agent",
            content=[Message(role="assistant", content=(
                "That's exciting! Good luck with your training."
            ))],
        ),
    ]
    print("\n  Mixed session (task + user info):")
    facts2 = await extract_memories(llm, mixed_events, existing=None)
    print(f"  Extracted facts ({len(facts2)}):")
    for f in facts2:
        print(f"    - {f}")
    user_facts = [f for f in facts2 if any(
        kw in f.lower() for kw in ["35", "berlin", "marathon", "training", "first"]
    ) and not any(
        kw in f.lower() for kw in ["kipchoge", "5.693", "2:53", "pace"]
    )]
    print(f"\n  User-relevant facts: {len(user_facts)}/{len(facts2)}")


# ── Section E — user isolation ─────────────────────────────────────────────────


async def section_e() -> None:
    _hr("Section E — User isolation: two users, no cross-contamination")

    mem = _fresh_memory("e")
    user_alice = "alice_iso"
    user_bob = "bob_iso"

    mem.add(user_id=user_alice, text="Alice's secret: she prefers cats over dogs.")
    mem.add(user_id=user_alice, text="Alice works at TechCorp as a senior engineer.")
    mem.add(user_id=user_bob, text="Bob's secret: he is allergic to peanuts.")
    mem.add(user_id=user_bob, text="Bob lives in Tokyo.")

    print(f"\n  Stored: 2 memories for {user_alice}, 2 for {user_bob}")

    # Alice searches
    alice_results = mem.search(user_id=user_alice, query="secrets and preferences", top_k=5, min_score=0.0)
    # Bob searches
    bob_results = mem.search(user_id=user_bob, query="secrets and preferences", top_k=5, min_score=0.0)

    print(f"\n  Alice sees ({len(alice_results)} records):")
    for r in alice_results:
        print(f"    score={r.score:.3f}  \"{r.text}\"")

    print(f"\n  Bob sees ({len(bob_results)} records):")
    for r in bob_results:
        print(f"    score={r.score:.3f}  \"{r.text}\"")

    alice_leaked = any("bob" in r.text.lower() or "peanut" in r.text.lower() or "tokyo" in r.text.lower()
                       for r in alice_results)
    bob_leaked = any("alice" in r.text.lower() or "cats" in r.text.lower() or "techcorp" in r.text.lower()
                     for r in bob_results)

    print(f"\n  Alice isolation: {'FAIL (Bob data visible!)' if alice_leaked else 'PASS'}")
    print(f"  Bob isolation  : {'FAIL (Alice data visible!)' if bob_leaked else 'PASS'}")

    # delete_user
    n = mem.delete_user(user_alice)
    print(f"\n  delete_user('{user_alice}'): {n} deleted")
    print(f"  list_all('{user_alice}') after delete: {len(mem.list_all(user_alice))} records")
    print(f"  list_all('{user_bob}') after delete: {len(mem.list_all(user_bob))} records (unchanged)")


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Block 14 long-term memory experiments")
    parser.add_argument(
        "--section", default="all",
        choices=["a", "b", "c", "d", "e", "all"],
    )
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[ERROR] ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def _run_all() -> None:
        if args.section in ("a", "all"):
            await section_a()
        if args.section in ("b", "all"):
            await section_b()
        if args.section in ("c", "all"):
            await section_c()
        if args.section in ("d", "all"):
            await section_d()
        if args.section in ("e", "all"):
            await section_e()

    try:
        asyncio.run(_run_all())
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
