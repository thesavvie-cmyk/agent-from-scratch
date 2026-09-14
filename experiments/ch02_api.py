"""Chapter 2 demo: Anthropic SDK vs LiteLLM, Conversation, extract, ask_many."""
from __future__ import annotations

import asyncio
import time

import anthropic
import litellm
from pydantic import BaseModel

from agentkit.basics import Conversation, ask, ask_many, extract
from agentkit.config import FAST_MODEL

litellm.suppress_debug_info = True

MAX_TOKENS = 64
QUESTION = "What is the capital of France? One word only."


def sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ── A. Raw response format comparison ────────────────────────
sep("A. Anthropic SDK vs LiteLLM — raw response structure")

# SDK uses bare model id (no provider prefix)
sdk_model_id = FAST_MODEL.split("/", 1)[-1]
sdk_client = anthropic.Anthropic()
sdk_resp = sdk_client.messages.create(
    model=sdk_model_id,
    max_tokens=MAX_TOKENS,
    messages=[{"role": "user", "content": QUESTION}],
)
print(f"\n[SDK]  type    : {type(sdk_resp).__name__}")
print(f"[SDK]  content : {sdk_resp.content}")
print(f"[SDK]  usage   : input_tokens={sdk_resp.usage.input_tokens}"
      f"  output_tokens={sdk_resp.usage.output_tokens}")

ll_resp = litellm.completion(
    model=FAST_MODEL,
    messages=[{"role": "user", "content": QUESTION}],
    max_tokens=MAX_TOKENS,
)
print(f"\n[LiteLLM]  type    : {type(ll_resp).__name__}")
print(f"[LiteLLM]  choices : {ll_resp.choices}")
print(f"[LiteLLM]  usage   : prompt_tokens={ll_resp.usage.prompt_tokens}"
      f"  completion_tokens={ll_resp.usage.completion_tokens}")

# ── B. Stateless vs Conversation ─────────────────────────────
sep("B. Stateless (no memory) vs Conversation (with memory)")

r1 = ask("My name is Kirill.")
r2 = ask("What is my name?")
print(f"\n[Stateless] turn 1 : {r1!r}")
print(f"[Stateless] turn 2 : {r2!r}  ← no memory")

conv = Conversation()
conv.send("My name is Kirill.")
r_conv = conv.send("What is my name?")
print(f"\n[Conversation]      : {r_conv!r}  ← remembers")

# ── C. Token growth over 4 turns ─────────────────────────────
sep("C. Conversation — input_tokens growth over 4 turns")

conv2 = Conversation(system="You are a helpful assistant. Keep replies under 20 words.")
turns = [
    "Tell me one fact about Mercury.",
    "Now one fact about Venus.",
    "What about Earth?",
    "And Mars?",
]
for turn in turns:
    reply = conv2.send(turn)
    tokens = conv2.last_usage.prompt_tokens if conv2.last_usage else "?"
    print(f"  [{tokens:>4} input tokens]  {reply[:70]}")

# ── D. Structured output ──────────────────────────────────────
sep("D. extract — structured output via response_format")


class ExtractedInfo(BaseModel):
    name: str
    email: str
    phone: str | None = None


sample = "Hi, I'm Bob Smith. Email: bob@example.com, phone: +1-800-555-0199."
info = extract(sample, schema=ExtractedInfo)
print(f"\nInput : {sample}")
print(f"name  : {info.name}")
print(f"email : {info.email}")
print(f"phone : {info.phone}")

# ── E. Sequential vs ask_many ─────────────────────────────────
sep("E. ask_many — sequential vs concurrent (10 prompts)")

questions = [
    f"What is {a} × {b}? One number only."
    for a, b in [(2, 3), (4, 5), (6, 7), (8, 9), (3, 3),
                 (7, 8), (5, 6), (9, 2), (4, 4), (6, 6)]
]

t0 = time.perf_counter()
seq_results = [ask(q) for q in questions]
t_seq = time.perf_counter() - t0

t0 = time.perf_counter()
par_results = asyncio.run(ask_many(questions, concurrency=5))
t_par = time.perf_counter() - t0

speedup = t_seq / t_par if t_par > 0 else 0
print(f"\nSequential : {t_seq:.2f}s")
print(f"Concurrent : {t_par:.2f}s  (×{speedup:.1f} faster)")
print(f"\nFirst 3 sequential : {seq_results[:3]}")
print(f"First 3 parallel   : {par_results[:3]}")
