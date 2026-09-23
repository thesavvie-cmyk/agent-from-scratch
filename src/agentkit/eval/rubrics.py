"""Evaluation rubrics (block 24).

Five rubrics:
  answer_relevance    — book rubric 1: does the answer address the question?
  source_credibility  — book rubric 2: are sources cited and credible?
  format_compliance   — book rubric 3: is the output format correct?
  trajectory_soundness — custom: specific facts must come from tool calls, not memory.
  data_provenance     — custom: numbers/stats in the answer must match tool output.

Rubric design notes
-------------------
- Each rubric has explicit PASS and FAIL criteria to minimise judge ambiguity.
- Examples are short (one PASS, one FAIL) so the judge prompt stays compact.
- trajectory_soundness and data_provenance require the trace summary to judge.
  The judge prompt must include TraceFeatures.summary_text().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Rubric:
    """A single evaluation criterion."""

    name: str
    description: str
    pass_criterion: str
    fail_criterion: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    requires_trace: bool = False   # true for trajectory-based rubrics


# ── Book rubrics ──────────────────────────────────────────────────────────────

ANSWER_RELEVANCE = Rubric(
    name="answer_relevance",
    description=(
        "The response directly addresses the user's question. "
        "It should answer what was asked, not a related but different question."
    ),
    pass_criterion=(
        "The response answers the exact question asked. It may include additional "
        "context but the core answer is present and on-topic."
    ),
    fail_criterion=(
        "The response answers a different question, is completely off-topic, "
        "says it cannot help without attempting the task, or is empty."
    ),
    examples=[
        {
            "input": "What is the boiling point of water in Fahrenheit?",
            "output": "Water boils at 212°F at sea level.",
            "verdict": "PASS",
            "reason": "Directly answers the question.",
        },
        {
            "input": "What is the boiling point of water in Fahrenheit?",
            "output": "Water is a very important molecule consisting of H2O.",
            "verdict": "FAIL",
            "reason": "Does not answer the question about boiling point.",
        },
    ],
)

SOURCE_CREDIBILITY = Rubric(
    name="source_credibility",
    description=(
        "When the agent cites external information, sources are named or "
        "discoverable. Claims about current/recent facts include a reference "
        "to where the information was found."
    ),
    pass_criterion=(
        "Factual claims include a source (URL, publication, or search result "
        "snippet). The agent attributes information or cites the search tool "
        "output. Statements clearly marked as general knowledge are acceptable."
    ),
    fail_criterion=(
        "Recent, specific, or numeric facts are stated without any attribution. "
        "The agent presents up-to-date statistics or current event information "
        "as if it is common knowledge without citing a source."
    ),
    examples=[
        {
            "input": "What is the population of Japan?",
            "output": (
                "According to a recent web search, Japan's population is "
                "approximately 125 million (source: World Bank data)."
            ),
            "verdict": "PASS",
            "reason": "Source is attributed.",
        },
        {
            "input": "What is the population of Japan?",
            "output": "Japan has a population of 125 million people.",
            "verdict": "FAIL",
            "reason": (
                "Specific current statistic given without any source attribution."
            ),
        },
    ],
)

FORMAT_COMPLIANCE = Rubric(
    name="format_compliance",
    description=(
        "The response follows any explicit formatting instructions in the task. "
        "If no format is specified, the response uses a clear, readable structure."
    ),
    pass_criterion=(
        "The response matches any explicit format instruction (e.g., 'give only "
        "the number', 'one item per line', 'JSON only'). If no format was "
        "requested, the response is clearly organised."
    ),
    fail_criterion=(
        "The response ignores an explicit format instruction — e.g., gives a "
        "paragraph when a single number was requested, or gives a number when "
        "a list was requested."
    ),
    examples=[
        {
            "input": "What is 15% of 847? Give a numeric answer only.",
            "output": "127.05",
            "verdict": "PASS",
            "reason": "Numeric answer only, as instructed.",
        },
        {
            "input": "What is 15% of 847? Give a numeric answer only.",
            "output": "15% of 847 is 127.05, which equals 127 dollars and 5 cents.",
            "verdict": "FAIL",
            "reason": "Extra prose given when only a number was requested.",
        },
    ],
)

# ── Custom rubrics ────────────────────────────────────────────────────────────

TRAJECTORY_SOUNDNESS = Rubric(
    name="trajectory_soundness",
    description=(
        "Specific facts, statistics, and current-events claims in the response "
        "must have been obtained through tool calls (search, code, file read). "
        "If the trace shows no relevant tool calls, the agent is answering from "
        "parametric memory — which is unreliable for current/specific facts."
    ),
    pass_criterion=(
        "If the response contains specific facts (names, numbers, dates, "
        "statistics, current events), the trace shows at least one relevant "
        "tool call before the answer. General knowledge and reasoning steps "
        "do not require tool calls."
    ),
    fail_criterion=(
        "The response contains specific current facts, exact statistics, or "
        "recent event claims AND the trace shows no relevant tool calls — the "
        "facts came from the model's training data without verification."
    ),
    examples=[
        {
            "input": "What is the current population of Tokyo?",
            "output": "Tokyo has approximately 13.96 million people in the city proper.",
            "trace": "search_calls=2, search_web×2",
            "verdict": "PASS",
            "reason": "Specific statistic backed by two search calls.",
        },
        {
            "input": "What is the current population of Tokyo?",
            "output": "Tokyo has approximately 13.96 million people in the city proper.",
            "trace": "search_calls=0, tool_calls=0",
            "verdict": "FAIL",
            "reason": (
                "Exact current statistic given but no search was performed — "
                "answer came from training data."
            ),
        },
    ],
    requires_trace=True,
)

DATA_PROVENANCE = Rubric(
    name="data_provenance",
    description=(
        "Numeric values, calculations, and specific measurements in the response "
        "must match what the tools actually returned. This catches the case where "
        "the agent ignores a tool result and uses a different number from memory "
        "or makes an arithmetic error that contradicts the code output."
    ),
    pass_criterion=(
        "Numbers and calculations in the response are consistent with what the "
        "trace shows the tools returned. If no tool was used for a calculation, "
        "the arithmetic is independently verifiable and correct."
    ),
    fail_criterion=(
        "The response states a number that differs from what the tool (code, "
        "calculator, search) returned — e.g., the agent ran code that computed X "
        "but reported a different value, or used a figure not in any search "
        "result."
    ),
    examples=[
        {
            "input": (
                "Kipchoge ran a marathon at 2:01:39 pace. How many hours to "
                "cover 363,104 km?"
            ),
            "output": "It would take approximately 17,050 hours.",
            "trace": "execute_python×1, code returned 17050.3",
            "verdict": "PASS",
            "reason": "Reported value matches code output.",
        },
        {
            "input": (
                "Kipchoge ran a marathon at 2:01:39 pace. How many hours to "
                "cover 363,104 km?"
            ),
            "output": "It would take approximately 18,000 hours.",
            "trace": "execute_python×1, code returned 17050.3",
            "verdict": "FAIL",
            "reason": "18,000 does not match the 17,050 the code returned.",
        },
    ],
    requires_trace=True,
)


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_RUBRICS: dict[str, Rubric] = {
    "answer_relevance": ANSWER_RELEVANCE,
    "source_credibility": SOURCE_CREDIBILITY,
    "format_compliance": FORMAT_COMPLIANCE,
    "trajectory_soundness": TRAJECTORY_SOUNDNESS,
    "data_provenance": DATA_PROVENANCE,
}

DEFAULT_RUBRICS: list[Rubric] = [
    ANSWER_RELEVANCE,
    SOURCE_CREDIBILITY,
    FORMAT_COMPLIANCE,
    TRAJECTORY_SOUNDNESS,
    DATA_PROVENANCE,
]
