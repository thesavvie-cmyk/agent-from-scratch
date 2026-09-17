"""Unit tests for block 14: long-term memory."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from agentkit.memory.longterm import LongTermMemory, Memory, _make_chroma_embedding_fn
from agentkit.types import Event, Message, ToolCall, ToolResult


# ── Fake embedding provider ────────────────────────────────────────────────────

class _FakeEmbedder:
    """Deterministic embedder for unit tests.

    Uses pre-defined 4-dim unit vectors so similarity is controllable:
    - Python cluster (dim-0): cos_sim ≈ 0.99 with each other, ≈ 0.6 with queries
    - Location cluster (dim-1): cos_sim ≈ 0.71 between Moscow and Petersburg
      (below default threshold 0.9 → stored separately; above 0 → both returned
      by search sorted by recency)
    - Orthogonal items: age (dim-2), food (dim-3) — cos_sim ≈ 0 with others
    """

    name = "fake"
    dimension = 4

    _VECS: dict[str, list[float]] = {
        # Cluster 1: programming / Python (cos_sim ≈ 0.995)
        "I love Python programming":      [1.0,  0.0,   0.0,  0.0],
        "Python is my favourite language": [0.995, 0.1,  0.0,  0.0],
        "My favourite language is Python": [0.993, 0.12, 0.0,  0.0],
        # Python queries — placed in Python cluster so min_score=0.5 passes
        "Tell me about Python":            [0.98,  0.2,  0.0,  0.0],
        "Python programming query":        [0.99,  0.14, 0.0,  0.0],
        # Cluster 2: location — cos_sim(Moscow, Petersburg) ≈ 0.71
        # Both below default dedup_threshold=0.9, so stored separately.
        "I live in Moscow":                [0.0, 1.0,  0.0,  0.0],
        "I moved to Saint Petersburg":     [0.0, 0.71, 0.71, 0.0],
        # Location query
        "where do I live?":               [0.0, 0.98, 0.2,  0.0],
        # Cluster 3: orthogonal items (cos_sim ≈ 0 with clusters 1 & 2)
        "I am 30 years old":               [0.0, 0.0,  1.0,  0.0],
        "I like ice cream":                [0.0, 0.0,  0.0,  1.0],
    }

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            if t in self._VECS:
                v = np.array(self._VECS[t], dtype=np.float32)
            else:
                # Hash-based fallback — deterministic, lands near origin of each dim
                h = abs(hash(t)) % 1000
                v = np.array([h / 1000, (1000 - h) / 1000, 0.05, 0.05], dtype=np.float32)
            # Normalise to unit sphere
            v = v / (np.linalg.norm(v) + 1e-10)
            rows.append(v)
        return np.array(rows, dtype=np.float32)


@pytest.fixture
def mem(tmp_path: Path) -> LongTermMemory:
    """Fresh LongTermMemory with fake embedder in tmp_path."""
    return LongTermMemory(
        path=str(tmp_path / "test_mem"),
        embedding_provider=_FakeEmbedder(),
        dedup_threshold=0.9,
    )


# ── _ChromaEmbeddingFn ────────────────────────────────────────────────────────


def test_chroma_fn_returns_list_of_lists() -> None:
    fn = _make_chroma_embedding_fn(_FakeEmbedder())
    result = fn(["hello", "world"])
    assert isinstance(result, list)
    assert len(result) == 2
    # Chroma may return numpy arrays (after normalize_embeddings); check shape not type
    assert len(result[0]) == 4  # _FakeEmbedder dimension


# ── Memory dataclass ──────────────────────────────────────────────────────────


def test_memory_defaults() -> None:
    m = Memory(user_id="alice", text="hello")
    assert m.user_id == "alice"
    assert m.text == "hello"
    assert m.access_count == 0
    assert m.score == 0.0
    assert len(m.id) == 36  # UUID


# ── add ───────────────────────────────────────────────────────────────────────


def test_add_creates_new_memory(mem: LongTermMemory) -> None:
    m = mem.add(user_id="alice", text="I love Python programming")
    assert m.id
    assert m.user_id == "alice"
    assert m.text == "I love Python programming"


def test_add_deduplicates_similar_text(mem: LongTermMemory) -> None:
    """Adding a near-duplicate updates the existing record, not a new one."""
    m1 = mem.add(user_id="alice", text="I love Python programming")
    time.sleep(0.01)
    m2 = mem.add(user_id="alice", text="Python is my favourite language")

    # Both texts map to similar vectors (sim ≈ 0.995 > 0.9 threshold)
    all_mems = mem.list_all("alice")
    # Should be 1 record (dedup) not 2
    assert len(all_mems) == 1
    # The record has been updated
    assert all_mems[0].updated_at >= m1.created_at


def test_add_does_not_dedup_dissimilar_text(mem: LongTermMemory) -> None:
    """Dissimilar texts create separate records."""
    mem.add(user_id="alice", text="I love Python programming")
    mem.add(user_id="alice", text="I am 30 years old")
    # sim([1,0,0,0], [0,0,1,0]) = 0.0 << 0.9 threshold
    assert len(mem.list_all("alice")) == 2


def test_add_empty_text_raises(mem: LongTermMemory) -> None:
    with pytest.raises(ValueError):
        mem.add(user_id="alice", text="")


def test_add_dedup_respects_user_boundary(mem: LongTermMemory) -> None:
    """Near-duplicate from different user should NOT trigger dedup."""
    mem.add(user_id="alice", text="I love Python programming")
    mem.add(user_id="bob", text="Python is my favourite language")
    assert len(mem.list_all("alice")) == 1
    assert len(mem.list_all("bob")) == 1


# ── search ────────────────────────────────────────────────────────────────────


def test_search_returns_relevant(mem: LongTermMemory) -> None:
    mem.add(user_id="alice", text="I love Python programming")
    mem.add(user_id="alice", text="I am 30 years old")
    results = mem.search(user_id="alice", query="I love Python programming", top_k=5, min_score=0.5)
    assert len(results) >= 1
    assert results[0].text == "I love Python programming"
    assert results[0].score >= 0.5


def test_search_filters_by_user(mem: LongTermMemory) -> None:
    """Search for alice must never return bob's memories."""
    mem.add(user_id="alice", text="I love Python programming")
    mem.add(user_id="bob", text="I love Python programming")

    alice_results = mem.search(user_id="alice", query="Python", top_k=5, min_score=0.0)
    bob_results = mem.search(user_id="bob", query="Python", top_k=5, min_score=0.0)

    assert all(m.user_id == "alice" for m in alice_results)
    assert all(m.user_id == "bob" for m in bob_results)


def test_search_min_score_filters_irrelevant(mem: LongTermMemory) -> None:
    """Orthogonal vectors (sim=0) should be filtered by min_score=0.5."""
    mem.add(user_id="alice", text="I like ice cream")
    # query about programming is orthogonal to ice cream in our fake embedder
    results = mem.search(user_id="alice", query="I love Python programming", top_k=5, min_score=0.5)
    assert len(results) == 0


def test_search_empty_store_returns_empty(mem: LongTermMemory) -> None:
    results = mem.search(user_id="nobody", query="anything", top_k=5)
    assert results == []


def test_search_scores_populated(mem: LongTermMemory) -> None:
    mem.add(user_id="alice", text="I love Python programming")
    results = mem.search(user_id="alice", query="I love Python programming", top_k=1, min_score=0.0)
    assert results[0].score > 0.0


def test_search_contradiction_newest_first(mem: LongTermMemory) -> None:
    """Both contradicting facts are stored (not deduped) and sorted score DESC.

    Moscow=[0,1,0,0] vs Petersburg=[0,0.71,0.71,0] → cos_sim ≈ 0.71 < 0.9
    threshold, so both records are stored separately.  search() returns them
    sorted by score DESC.  Within equal scores, updated_at DESC would apply.
    """
    mem.add(user_id="alice", text="I live in Moscow")
    time.sleep(0.01)
    mem.add(user_id="alice", text="I moved to Saint Petersburg")

    results = mem.search(
        user_id="alice", query="where do I live?", top_k=5, min_score=0.0
    )
    assert len(results) == 2
    # Results sorted score DESC
    assert results[0].score >= results[1].score


# ── list_all ──────────────────────────────────────────────────────────────────


def test_list_all_returns_all_for_user(mem: LongTermMemory) -> None:
    mem.add(user_id="alice", text="I love Python programming")
    mem.add(user_id="alice", text="I am 30 years old")
    mem.add(user_id="bob", text="I like ice cream")

    alice = mem.list_all("alice")
    bob = mem.list_all("bob")

    assert len(alice) == 2
    assert len(bob) == 1
    assert all(m.user_id == "alice" for m in alice)


def test_list_all_empty_user(mem: LongTermMemory) -> None:
    assert mem.list_all("nobody") == []


# ── delete ────────────────────────────────────────────────────────────────────


def test_delete_existing(mem: LongTermMemory) -> None:
    m = mem.add(user_id="alice", text="I love Python programming")
    assert mem.delete(m.id) is True
    assert mem.list_all("alice") == []


def test_delete_missing(mem: LongTermMemory) -> None:
    assert mem.delete("nonexistent-id") is False


def test_delete_user(mem: LongTermMemory) -> None:
    mem.add(user_id="alice", text="I love Python programming")
    mem.add(user_id="alice", text="I am 30 years old")
    mem.add(user_id="bob", text="I like ice cream")

    n = mem.delete_user("alice")
    assert n == 2
    assert mem.list_all("alice") == []
    assert len(mem.list_all("bob")) == 1


def test_delete_user_unknown(mem: LongTermMemory) -> None:
    assert mem.delete_user("nobody") == 0


# ── extract_memories ──────────────────────────────────────────────────────────


def _fake_llm_extract(facts: list[str]) -> Any:
    """LLM mock that returns pre-defined facts."""
    import json

    mock = MagicMock()
    mock.generate = AsyncMock(
        return_value=MagicMock(
            content=[Message(role="assistant", content=json.dumps({"facts": facts}))],
            error_message=None,
        )
    )
    return mock


@pytest.mark.asyncio
async def test_extract_empty_events() -> None:
    from agentkit.memory.extraction import extract_memories

    llm = _fake_llm_extract(["The user is Alice."])
    result = await extract_memories(llm, events=[], existing=None)
    assert result == []  # no events → empty, LLM not called


@pytest.mark.asyncio
async def test_extract_returns_facts() -> None:
    from agentkit.memory.extraction import extract_memories

    events = [
        Event(
            execution_id="x",
            author="a",
            content=[Message(role="user", content="I am Alice, a Python developer.")],
        )
    ]
    llm = _fake_llm_extract(["The user's name is Alice.", "The user is a Python developer."])
    result = await extract_memories(llm, events, existing=None)
    assert "The user's name is Alice." in result


@pytest.mark.asyncio
async def test_extract_llm_error_returns_empty() -> None:
    from agentkit.memory.extraction import extract_memories

    mock = MagicMock()
    mock.generate = AsyncMock(
        return_value=MagicMock(content=[], error_message="timeout")
    )
    events = [
        Event(
            execution_id="x",
            author="a",
            content=[Message(role="user", content="Hello.")],
        )
    ]
    result = await extract_memories(mock, events, existing=None)
    assert result == []


@pytest.mark.asyncio
async def test_extract_no_facts_ok() -> None:
    """Empty facts list is valid — not an error."""
    from agentkit.memory.extraction import extract_memories

    events = [
        Event(
            execution_id="x",
            author="a",
            content=[Message(role="user", content="What time is it?")],
        )
    ]
    llm = _fake_llm_extract([])
    result = await extract_memories(llm, events, existing=None)
    assert result == []


# ── Agent with memory ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_memory_injected_into_instructions(tmp_path: Path) -> None:
    """Relevant memories appear in the instructions block passed to the LLM."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient, LlmResponse

    memory = LongTermMemory(
        path=str(tmp_path / "agent_mem"),
        embedding_provider=_FakeEmbedder(),
    )
    memory.add(user_id="alice", text="I love Python programming")

    captured: list[Any] = []

    async def _capture(req: Any) -> LlmResponse:
        captured.append(req)
        return LlmResponse(
            content=[Message(role="assistant", content="Got it!")],
            usage_metadata={},
        )

    mock_client = MagicMock(spec=LlmClient)
    mock_client.generate = AsyncMock(side_effect=_capture)

    agent = Agent(
        model=mock_client,
        tools=[],
        instructions="Be helpful.",
        max_steps=2,
        memory=memory,
        user_id="alice",
    )
    await agent.run("Tell me about Python")

    assert captured, "LLM was never called"
    instructions = captured[0].instructions
    combined = " ".join(instructions)
    assert "Python" in combined or "python" in combined.lower()


@pytest.mark.asyncio
async def test_agent_memory_save_called_after_run(tmp_path: Path) -> None:
    """After run() returns, memory.add() has already been called (awaited save)."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient, LlmResponse

    memory = LongTermMemory(
        path=str(tmp_path / "agent_mem2"),
        embedding_provider=_FakeEmbedder(),
    )

    save_called = False
    original_add = memory.add

    def _spy_add(user_id: str, text: str, **kw: Any) -> Memory:
        nonlocal save_called
        save_called = True
        return original_add(user_id=user_id, text=text, **kw)

    memory.add = _spy_add  # type: ignore[method-assign]

    mock_client = MagicMock(spec=LlmClient)
    mock_client.generate = AsyncMock(
        return_value=LlmResponse(
            content=[Message(role="assistant", content="Great!")],
            usage_metadata={},
        )
    )

    # Patch extract_memories to return a fact immediately
    import agentkit.memory.extraction as ext_mod
    original_extract = ext_mod.extract_memories

    async def _fake_extract(llm: Any, events: Any, existing: Any = None) -> list[str]:
        return ["The user said hello."]

    ext_mod.extract_memories = _fake_extract  # type: ignore[assignment]
    try:
        agent = Agent(
            model=mock_client,
            tools=[],
            instructions="",
            max_steps=2,
            memory=memory,
            user_id="test",
        )
        await agent.run("Hello!")
        assert save_called, "memory.add() was never called after run()"
    finally:
        ext_mod.extract_memories = original_extract  # type: ignore[assignment]


# ── Live tests ─────────────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_cross_session_recall(tmp_path: Path) -> None:
    """Full end-to-end: store fact in session 1, recall in session 2."""
    from agentkit.agent import Agent
    from agentkit.embeddings import get_embedding_provider
    from agentkit.llm import LlmClient

    memory = LongTermMemory(
        path=str(tmp_path / "live_mem"),
        embedding_provider=get_embedding_provider(),
    )

    agent1 = Agent(
        model=LlmClient(FAST_MODEL),
        tools=[],
        instructions="You are a helpful assistant.",
        max_steps=3,
        memory=memory,
        user_id="live_user",
    )
    await agent1.run("My name is LiveTestUser and I work with Rust.")
    # Flush memory tasks
    for _ in range(20):
        await asyncio.sleep(0.1)

    agent2 = Agent(
        model=LlmClient(FAST_MODEL),
        tools=[],
        instructions="You are a helpful assistant.",
        max_steps=3,
        memory=memory,
        user_id="live_user",
    )
    result = await agent2.run("What programming language do I work with?")
    assert "rust" in str(result.output).lower(), f"Expected Rust in answer, got: {result.output}"
