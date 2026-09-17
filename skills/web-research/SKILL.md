---
name: web-research
description: Systematic web research strategy — query decomposition, when snippets suffice vs full extraction, collecting structured data via code instead of agent round-trips
version: "1.0"
---

# Web Research

## When to search vs answer from memory

**Search** when the question requires:
- Events after your training cutoff
- Specific statistics, rankings, or figures that change over time
- Verification of surprising claims
- Lists of entities you need to enumerate (see "structured data" below)

**Do not search** for:
- Mathematical definitions, formulas, well-known constants
- Historical events that are stable and well-documented
- Reasoning steps — search for facts, then reason yourself

---

## Query strategy

1. **Broad first**: one query to confirm the topic exists and find the right framing
2. **Specific follow-up**: targeted queries for each sub-fact needed
3. **Verify on conflict**: if two snippets contradict, run a third tiebreaker query

For multi-part questions, break into sub-questions and search each separately.
Keep queries short (3–6 words) — long queries produce lower-quality snippets.

---

## When snippets suffice

Snippets (`max_results=5`, default) are enough when:
- The answer is a single fact (a name, a date, a number)
- The title clearly signals the right source
- Two or more snippets agree on the same value

**Stop after one search** when the snippet contains the answer. The most common
mistake is searching again "to be sure" when the answer is already there.

---

## When to collect data in code (the 46-cities lesson)

Some tasks require data for *many* entities — presidents' birthplaces, capitals
of 50 states, box-office grosses for 200 films. These defeat the agent loop:
- 46 agent rounds × LLM overhead = slow, expensive, and hits max_steps
- Reflection and planning did not help (blocks 15–16)
- Search-only approaches produced 0/46 correct entries at max_steps

**The solution**: run all searches inside a single `execute_python` call using
the env-bridge `search_web` function:

```python
presidents = {
    1: "George Washington", 2: "John Adams", 3: "Thomas Jefferson",
    # ... all 46
}
birthplaces = {}
for num, name in presidents.items():
    result = search_web(query=f"{name} president birthplace city state", max_results=3)
    # result is formatted text: "[1] Title\nURL\nSnippet\n\n[2] ..."
    first_snippet = result.split("\n\n")[0]
    birthplaces[num] = first_snippet

for num in sorted(birthplaces):
    print(f"{num}. {presidents[num]}: {birthplaces[num]}")
```

This approach:
- Runs all N searches in **one** agent step (1 LLM round-trip)
- Uses no extra token budget per search
- Produces structured, numbered output that is always complete
- Solved 46/46 presidents vs 0/46 for search-only agent (block 17)

**Threshold**: use code-loop when N > 5 entities need the same fact.

---

## Parsing search results

`search_web` returns formatted text:

```
[1] Title of result
https://example.com/page
Snippet text describing what the page says about the topic.

[2] Another result
...
```

Extract facts from snippets with simple string ops or regex:

```python
import re

# Extract a 4-digit year
m = re.search(r'\b(19|20)\d{2}\b', snippet)
year = m.group() if m else None

# Extract the first number
m = re.search(r'\b\d[\d,\.]*\b', snippet)
value = m.group().replace(',', '') if m else None
```

---

## Common pitfalls

| Pitfall | Fix |
|---------|-----|
| Searching twice for the same fact | Check if snippet already has the answer |
| Long verbose query → noisy results | Shorten to 3–6 keyword terms |
| Looping with 10+ separate agent calls | Use execute_python loop (see above) |
| Giving up after one failed search | Rephrase: use synonyms, add context |
| Treating a snippet as a citation | Cross-verify for numbers > 5% off expected |
