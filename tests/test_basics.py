from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import litellm
import pytest
from pydantic import BaseModel

from agentkit.basics import Conversation, StructuredOutputError, ask_many, extract


# ── helpers ──────────────────────────────────────────────────────────────────

def _resp(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = 10
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ── Conversation ──────────────────────────────────────────────────────────────

def test_conversation_accumulates_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = ["Hello!", "It's Monday.", "Goodbye!"]
    idx = 0

    def fake_completion(**kwargs: object) -> MagicMock:
        nonlocal idx
        r = _resp(replies[idx])
        idx += 1
        return r

    monkeypatch.setattr(litellm, "completion", fake_completion)

    conv = Conversation(system="Be helpful.")
    conv.send("Hi")
    conv.send("What day is it?")
    conv.send("Bye")

    history = conv.messages
    assert len(history) == 6
    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]


def test_conversation_system_is_first(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, str]] = []

    def fake_completion(**kwargs: object) -> MagicMock:
        captured.extend(kwargs["messages"])  # type: ignore[arg-type]
        return _resp("ok")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    conv = Conversation(system="Be concise.")
    conv.send("Hello")

    assert captured[0]["role"] == "system"
    assert captured[0]["content"] == "Be concise."
    assert captured[1]["role"] == "user"


# ── extract ───────────────────────────────────────────────────────────────────

class _Info(BaseModel):
    name: str
    email: str


def test_extract_valid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kw: _resp('{"name": "Alice", "email": "alice@example.com"}'),
    )
    result = extract("Contact Alice at alice@example.com", _Info)
    assert isinstance(result, _Info)
    assert result.name == "Alice"
    assert result.email == "alice@example.com"


def test_extract_invalid_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kw: _resp("not json at all"),
    )
    with pytest.raises(StructuredOutputError) as exc_info:
        extract("some text", _Info)
    assert "not json at all" in exc_info.value.raw


# ── ask_many ──────────────────────────────────────────────────────────────────

async def test_ask_many_respects_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    max_active = 0

    async def fake_acompletion(**kwargs: object) -> MagicMock:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return _resp("ok")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    results = await ask_many([f"q{i}" for i in range(10)], concurrency=3)

    assert max_active <= 3
    assert len(results) == 10
    assert all(r == "ok" for r in results)


async def test_ask_many_exception_preserved_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(*, messages: list[dict[str, str]], **kwargs: object) -> MagicMock:
        content = messages[0]["content"]
        if content == "fail":
            raise ValueError("boom")
        return _resp(f"ok:{content}")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    results = await ask_many(["a", "b", "fail", "c", "d"], concurrency=5)

    assert len(results) == 5
    assert results[0] == "ok:a"
    assert results[1] == "ok:b"
    assert isinstance(results[2], ValueError)
    assert results[3] == "ok:c"
    assert results[4] == "ok:d"


# ── live tests ────────────────────────────────────────────────────────────────

@pytest.mark.live
def test_ask_live_nonempty() -> None:
    from agentkit.basics import ask

    result = ask("What is 2+2? Reply with just the number.")
    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.live
def test_conversation_live_remembers_name() -> None:
    conv = Conversation()
    conv.send("My name is Kirill. Just say 'Got it'.")
    reply = conv.send("What is my name?")
    assert "kirill" in reply.lower()


@pytest.mark.live
def test_extract_live() -> None:
    class ExtractedInfo(BaseModel):
        name: str
        email: str
        phone: str | None = None

    text = "Hi, I'm Alice. Reach me at alice@example.com or +1-555-0100."
    result = extract(text, ExtractedInfo)
    assert "Alice" in result.name
    assert "alice@example.com" in result.email
    assert result.phone is not None


@pytest.mark.live
async def test_ask_many_live_no_exceptions() -> None:
    prompts = [f"What is {i} + 1? Reply with just the number." for i in range(10)]
    results = await ask_many(prompts, concurrency=5)
    assert len(results) == 10
    assert all(isinstance(r, str) for r in results)
