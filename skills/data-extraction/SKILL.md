---
name: data-extraction
description: Collect and structure data from multiple web sources using Python loops — for tasks requiring counts, rankings, or comparisons across many entities
version: "1.0"
scripts:
  - scripts/extract_table.py
---

# Data Extraction

Use this skill when you need structured data across **multiple entities** —
not a single lookup, but a collection of facts that you then count, sort,
filter, or compare.

---

## Core pattern: search → collect → structure

```python
entities = ["Entity A", "Entity B", "Entity C", ...]  # your list

data = {}
errors = []
for entity in entities:
    result = search_web(query=f"{entity} the specific fact you need", max_results=3)
    if "(no results)" in result or not result.strip():
        errors.append(entity)
        data[entity] = None
    else:
        data[entity] = result.split("\n\n")[0]  # take first result block

# Quality check
print(f"Collected: {len(data) - len(errors)}/{len(entities)}")
print(f"Missing:   {errors[:10]}")
```

---

## When to use this skill

| Indicator | Use data-extraction |
|-----------|---------------------|
| "List all X that satisfy Y" | Yes |
| "How many X have property Y" | Yes |
| "What are the top N by some metric" | Yes |
| "What is the capital of France" | No — single lookup |
| "Summarize this article" | No — single source |

**Threshold**: if you need the same fact for more than 5 entities, use a loop.

---

## Parsing and normalising

Tavily returns formatted text. Extract values with regex or string ops:

```python
import re

def extract_year(text: str) -> str | None:
    m = re.search(r'\b(19|20)\d{2}\b', text)
    return m.group() if m else None

def extract_number(text: str) -> float | None:
    m = re.search(r'\b(\d[\d,]*(?:\.\d+)?)\b', text)
    if m:
        return float(m.group().replace(',', ''))
    return None
```

Always normalise before comparing:
```python
# Normalise names
import unicodedata
def norm(s: str) -> str:
    return unicodedata.normalize('NFC', s).lower().strip()
```

---

## Handling missing data

A missing entry is better than a wrong entry:

```python
missing = [e for e, v in data.items() if v is None]
if len(missing) > len(entities) * 0.1:  # > 10% missing
    print(f"WARNING: {len(missing)} entries missing — check query template")
    # Retry with a different query pattern
    for entity in missing[:3]:
        result = search_web(query=f"{entity} alternative query phrasing", max_results=5)
        if result and "(no results)" not in result:
            data[entity] = result.split("\n\n")[0]
```

---

## Output formatting

Produce **numbered output in code** — do not construct numbered lists manually:

```python
# Sorted by some attribute
ranked = sorted(
    [(k, v) for k, v in data.items() if v is not None],
    key=lambda x: x[1]
)
for i, (entity, value) in enumerate(ranked, 1):
    print(f"{i}. {entity}: {value}")
```

This guarantees correct numbering regardless of how many items were collected.

---

## Scripts in this skill

| Script | Purpose |
|--------|---------|
| `scripts/extract_table.py` | Extract structured data from a list and output as CSV |

Usage: `python /home/user/skills/data-extraction/scripts/extract_table.py`
