"""Memory extraction from agent events (block 14).

extract_memories() uses a cheap LLM call (FAST_MODEL) to identify stable
facts about the *user* from a session's event history.

What counts as a stable fact
-----------------------------
- Identity: name, age, location, profession
- Preferences: languages, tools, foods, hobbies, communication style
- Context: current project, team, company, role
- Goals or constraints that hold across sessions

What does NOT count
-------------------
- Content of the task being done (search results, calculations)
- Facts about third parties that are not about the user
- Information that is only true today / ephemeral
- Repetitions of facts already captured in *existing*

The function is resilient: if the LLM returns no facts, or fails, or the
JSON is malformed, it returns an empty list.  Callers must not treat an
empty result as an error.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from agentkit.llm import LlmClient
    from agentkit.types import Event

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = """\
You are a memory extraction assistant.  Your job is to identify stable,
long-term facts about the USER from a conversation transcript.

Rules:
1. Only include facts that describe the USER, not the task or third parties.
2. Only include facts that are stable across sessions (not "today I need X").
3. Do NOT include search results, document contents, or calculation outputs.
4. Do NOT duplicate facts already in the EXISTING list.
5. If there are no new stable facts, return an empty list — that is fine.
6. Return concise, self-contained statements (one fact per entry).
   Good: "The user's name is Alice."
   Good: "The user works as a data engineer."
   Bad:  "The user searched for marathon times." (task content)
   Bad:  "Kipchoge ran 2:01." (fact about a third party)
"""

_EXTRACT_PROMPT_TEMPLATE = """\
Conversation transcript:
{transcript}

{existing_block}

Extract stable facts about the user from the transcript above.
Return JSON: {{"facts": ["...", "..."]}}
If no new stable facts exist, return: {{"facts": []}}
"""


class ExtractedMemories(BaseModel):
    facts: list[str]


async def extract_memories(
    llm_client: LlmClient,
    events: list[Event],
    existing: list[str] | None = None,
) -> list[str]:
    """Extract stable user facts from *events* using a cheap LLM call.

    Parameters
    ----------
    llm_client:
        LLM client (will use FAST_MODEL for cost efficiency).
    events:
        Agent events from the session run to analyse.
    existing:
        Facts already stored for this user.  New facts that duplicate
        these are skipped.

    Returns
    -------
    list[str]
        Extracted facts.  Empty list if none found or on any error.
    """
    from agentkit.llm import LlmRequest
    from agentkit.types import Message, ToolCall, ToolResult

    if not events:
        return []

    # Build a readable transcript from events
    lines: list[str] = []
    for evt in events:
        for item in evt.content:
            if isinstance(item, Message) and item.role in ("user", "assistant"):
                lines.append(f"[{item.role}]: {item.content[:500]}")
            elif isinstance(item, ToolCall):
                lines.append(f"[tool_call]: {item.name}({str(item.arguments)[:200]})")
            elif isinstance(item, ToolResult) and item.status == "success":
                snippet = str(item.content)[:300]
                lines.append(f"[tool_result]: {snippet}")

    if not lines:
        return []

    transcript = "\n".join(lines)

    existing_block = ""
    if existing:
        existing_str = "\n".join(f"- {f}" for f in existing[:50])
        existing_block = f"Already known facts (do NOT duplicate):\n{existing_str}\n"

    prompt = _EXTRACT_PROMPT_TEMPLATE.format(
        transcript=transcript,
        existing_block=existing_block,
    )

    try:
        request = LlmRequest(
            instructions=[_EXTRACT_SYSTEM],
            contents=[Message(role="user", content=prompt)],
        )
        response = await llm_client.generate(request)

        if response.error_message:
            logger.warning("extract_memories LLM error: %s", response.error_message)
            return []

        # Find the text response
        text = ""
        for item in response.content:
            if isinstance(item, Message):
                text = item.content
                break

        if not text:
            return []

        # Parse JSON — be lenient: strip markdown fences and trailing content
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines()
                if not line.startswith("```")
            ).strip()
        # Truncate to first complete JSON object (handles LLM reasoning after JSON)
        brace = text.find("{")
        if brace != -1:
            depth = 0
            for i, ch in enumerate(text[brace:], start=brace):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[brace : i + 1]
                        break

        parsed = ExtractedMemories.model_validate_json(text)
        facts = [f.strip() for f in parsed.facts if f.strip()]
        return facts

    except Exception as exc:  # noqa: BLE001
        logger.warning("extract_memories failed: %s", exc)
        return []
