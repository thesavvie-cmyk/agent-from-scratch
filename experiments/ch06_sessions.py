"""Chapter 6 session experiments (block 13).

Sections
--------
a) Three dependent questions in one session — each answer builds on the last
b) Process restart simulation — create session, "remember X", reload, "what did I say?"
c) State persistence — remember_fact/recall_fact tools that write to session.state
d) Context growth — 10 turns with and without compaction (token comparison table)
e) User isolation — two users, verify sessions don't leak

Usage
-----
    uv run python experiments/ch06_sessions.py --section a
    uv run python experiments/ch06_sessions.py --section all

Requires ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import os

from agentkit.agent import Agent
from agentkit.config import FAST_MODEL
from agentkit.llm import LlmClient
from agentkit.memory.session import InMemorySessionStore, Session, SqliteSessionStore
from agentkit.tools.base import FunctionTool, tool

RESULTS_DIR = Path(__file__).parent.parent / "results"
SESSION_DB = RESULTS_DIR / "ch06_sessions.db"

# ── helpers ────────────────────────────────────────────────────────────────────


def _hr(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def _make_agent(store: Any = None, tools: list[Any] | None = None) -> Agent:
    return Agent(
        model=LlmClient(FAST_MODEL),
        tools=tools or [],
        instructions=(
            "You are a helpful assistant with memory across turns. "
            "Answer concisely."
        ),
        max_steps=4,
        session_store=store,
    )


# ── Section A — dependent questions ───────────────────────────────────────────


async def section_a() -> None:
    _hr("Section A — Three dependent questions in one session")

    store = InMemorySessionStore()
    agent = _make_agent(store)
    session = store.create(user_id="alice")

    questions = [
        "My name is Alice and I love Italian food. Remember that.",
        "What is my name?",
        "What kind of food do I love?",
    ]

    print()
    for q in questions:
        print(f"  You: {q}")
        result = await agent.run(q, session=session)
        # Reload session to reflect persisted events
        session = store.get(session.session_id) or session
        print(f"  Agent: {result.output}")
        print()

    print(f"  Session {session.session_id}: {len(session.events)} events total")


# ── Section B — restart simulation ────────────────────────────────────────────


async def section_b() -> None:
    _hr("Section B — Process restart simulation (SqliteSessionStore)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Clean up old db for clean demo
    if SESSION_DB.exists():
        SESSION_DB.unlink()

    store1 = SqliteSessionStore(str(SESSION_DB))
    agent1 = _make_agent(store1)
    session = store1.create(user_id="bob")
    sid = session.session_id

    print(f"\n  [process 1] store1 id={id(store1)}")
    print(f"  [process 1] session_id = {sid}")
    result = await agent1.run(
        "My secret number is 42. Remember it.", session=session
    )
    # Reload to confirm events were flushed to disk
    session = store1.get(sid) or session
    print(f"  [process 1] Agent: {result.output}")
    print(f"  [process 1] Events in DB after turn: {len(session.events)}")

    # Simulate process restart: completely new Python objects, no shared state.
    # The ONLY way store2 can find the session is by reading the SQLite file.
    del store1, agent1, session
    print("\n  [restart] del store1 / agent1 / session — all process-1 objects gone")

    store2 = SqliteSessionStore(str(SESSION_DB))
    agent2 = _make_agent(store2)
    print(f"  [process 2] store2 id={id(store2)}  (different object)")
    session2 = store2.get(sid)

    if session2 is None:
        print("  [FAIL] session not found after restart")
        return

    print(f"  [process 2] Loaded session with {len(session2.events)} past events from disk")
    result2 = await agent2.run("What was my secret number?", session=session2)
    print(f"  [process 2] Agent: {result2.output}")

    ok = "42" in str(result2.output)
    print(f"\n  Result: {'PASS — agent recalled the number from disk' if ok else 'FAIL — number not recalled'}")


# ── Section C — state persistence with tools ──────────────────────────────────


async def section_c() -> None:
    _hr("Section C — State persistence via remember_fact/recall_fact tools")

    store = InMemorySessionStore()
    session = store.create(user_id="carol")

    # Build tools that write/read session.state
    @tool
    def remember_fact(context: Any, key: str, value: str) -> str:
        """Store a fact in session state under *key*."""
        session_state = context.state.get("session_state", context.state)
        session_state[key] = value
        return f"Remembered: {key} = {value}"

    @tool
    def recall_fact(context: Any, key: str) -> str:
        """Retrieve a fact stored under *key* from session state."""
        session_state = context.state.get("session_state", context.state)
        return str(session_state.get(key, f"[no fact stored for '{key}']"))

    agent = _make_agent(store, tools=[remember_fact, recall_fact])

    print()
    result1 = await agent.run(
        "Use the remember_fact tool to store: favourite_colour = blue",
        session=session,
    )
    session = store.get(session.session_id) or session
    print(f"  Turn 1: {result1.output}")
    print(f"  Session state after turn 1: {session.state}")

    result2 = await agent.run(
        "Use the recall_fact tool to tell me my favourite_colour.",
        session=session,
    )
    session = store.get(session.session_id) or session
    print(f"  Turn 2: {result2.output}")
    print(f"  Session state after turn 2: {session.state}")

    ok = "blue" in str(result2.output).lower()
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")


# ── Section D — context growth with/without compaction ────────────────────────


async def section_d() -> None:
    _hr("Section D — Context growth across 10 turns (with vs without compaction)")

    from agentkit.memory.budget import ContextBudget
    from agentkit.memory.compaction import DropOldest, TruncateOldToolResults

    questions = [
        "What is 1 + 1?",
        "And if I double that?",
        "Now multiply by 3.",
        "Subtract 5.",
        "What was the very first answer you gave me?",
        "What is the square root of 16?",
        "Is that more or less than the running total?",
        "What was the result of multiplying by 3 earlier?",
        "Summarise all the numbers we've discussed so far.",
        "Which step produced the largest number?",
    ]

    async def _run_config(
        name: str, budget: ContextBudget | None
    ) -> tuple[str, list[int], list[int]]:
        from agentkit.memory.budget import estimate_context_tokens
        from agentkit.types import ContentItem

        store = InMemorySessionStore()
        agent = Agent(
            model=LlmClient(FAST_MODEL),
            tools=[],
            instructions="You are a helpful assistant. Answer concisely.",
            max_steps=3,
            session_store=store,
            context_budget=budget,
        )
        session = store.create(user_id="dave")
        token_snapshots: list[int] = []
        event_snapshots: list[int] = []

        for q in questions:
            await agent.run(q, session=session)
            session = store.get(session.session_id) or session

            all_contents: list[ContentItem] = []
            for evt in session.events:
                all_contents.extend(evt.content)  # type: ignore[arg-type]
            token_snapshots.append(estimate_context_tokens(all_contents))
            event_snapshots.append(len(session.events))

        return name, token_snapshots, event_snapshots

    # Strategy selection depends on what fills the context:
    #   - tool-heavy sessions: TruncateOldToolResults (keeps last N search results)
    #   - text-only sessions:  DropOldest (drops oldest messages until under budget)
    #
    # Sessions need *lower* thresholds than single-run contexts because history
    # accumulates permanently — it never resets between agent.run calls.
    # At ~43 tok/turn (text-only), max_tokens=200 fires around turn 5.
    # A real session with search results (~500-2000 tok each) would fill 200 tokens
    # after the very first tool call.
    budget = ContextBudget(
        max_tokens=200,
        strategies=[DropOldest(max_tokens=200)],
    )

    print()
    configs: list[tuple[str, ContextBudget | None]] = [
        ("no_budget", None),
        ("with_budget", budget),
    ]
    results: list[tuple[str, list[int], list[int]]] = []
    for name, b in configs:
        _, tok_snaps, evt_snaps = await _run_config(name, b)
        results.append((name, tok_snaps, evt_snaps))

    nb_tok, wb_tok = results[0][1], results[1][1]
    nb_evt = results[0][2]

    print(f"\n  {'Turn':>5}  {'Question (abbrev)':<32}  {'Events':>7}  {'no_budget':>10}  {'with_budget':>10}  {'saved%':>6}")
    print(f"  {'─'*5}  {'─'*32}  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*6}")
    for i, (q, a, b_val, e) in enumerate(
        zip(questions, nb_tok, wb_tok, nb_evt), 1
    ):
        pct = int(100 * (a - b_val) / a) if a else 0
        print(f"  {i:>5}  {q[:32]:<32}  {e:>7}  {a:>10,}  {b_val:>10,}  {pct:>5}%")

    final_nb, final_wb = nb_tok[-1], wb_tok[-1]
    saved = final_nb - final_wb
    print(f"\n  Final: no_budget={final_nb:,}  with_budget={final_wb:,}  "
          f"saved={saved:,} ({100*saved//max(final_nb,1)}%)")
    print(f"  Note: session history grows ~{final_nb // len(questions):,} tok/turn on average "
          f"(vs single-run context which resets each agent.run call)")


# ── Section E — user isolation ─────────────────────────────────────────────────


async def section_e() -> None:
    _hr("Section E — User isolation: two users, no cross-contamination")

    store = InMemorySessionStore()
    agent = _make_agent(store)

    # Alice's session — plant her secret
    session_alice = store.create(user_id="alice")
    await agent.run("My secret code is ALPHA-1.", session=session_alice)
    # Must reload: append_events has flushed new events; reload so next turn sees them
    session_alice = store.get(session_alice.session_id) or session_alice

    # Bob's session — plant his secret
    session_bob = store.create(user_id="bob")
    await agent.run("My secret code is BETA-2.", session=session_bob)
    session_bob = store.get(session_bob.session_id) or session_bob

    print(f"\n  Alice events: {len(session_alice.events)}  Bob events: {len(session_bob.events)}")
    print(f"  Alice session_id: {session_alice.session_id}")
    print(f"  Bob   session_id: {session_bob.session_id}")
    print(f"  IDs are different: {session_alice.session_id != session_bob.session_id}")

    # Alice asks — must see ALPHA-1, must NOT see BETA-2
    result_alice = await agent.run("What is my secret code?", session=session_alice)
    # Bob asks — must see BETA-2, must NOT see ALPHA-1
    result_bob = await agent.run("What is my secret code?", session=session_bob)

    alice_answer = str(result_alice.output)
    bob_answer = str(result_bob.output)

    print(f"\n  Alice's answer: {alice_answer}")
    print(f"  Bob's answer  : {bob_answer}")

    alice_ok = "alpha" in alice_answer.lower() and "beta" not in alice_answer.lower()
    bob_ok = "beta" in bob_answer.lower() and "alpha" not in bob_answer.lower()

    print(f"\n  Alice isolation: {'PASS' if alice_ok else 'FAIL'}")
    print(f"  Bob isolation  : {'PASS' if bob_ok else 'FAIL'}")

    # Verify list_sessions returns separate lists
    alice_sessions = store.list_sessions("alice")
    bob_sessions = store.list_sessions("bob")
    print(f"\n  store.list_sessions('alice'): {len(alice_sessions)} session(s)")
    print(f"  store.list_sessions('bob'):   {len(bob_sessions)} session(s)")

    # delete_user
    deleted = store.delete_user("alice")
    print(f"\n  delete_user('alice'): {deleted} deleted")
    print(f"  list_sessions('alice') after delete: {len(store.list_sessions('alice'))}")


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Block 13 session experiments")
    parser.add_argument(
        "--section",
        default="all",
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
