#!/usr/bin/env python3
"""Extract structured data from a list of entities and output as CSV.

This script demonstrates the core data-extraction pattern:
  entity list → search loop → structured CSV output

Edit ENTITIES and QUERY_TEMPLATE before running.

Usage: python extract_table.py [output.csv]
"""
import csv
import re
import sys


# ── Configure these ────────────────────────────────────────────────────────────

ENTITIES = [
    "George Washington",
    "John Adams",
    "Thomas Jefferson",
    "James Madison",
    "James Monroe",
]

QUERY_TEMPLATE = "{entity} birthplace city state"

OUTPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "output.csv"


# ── Helpers ────────────────────────────────────────────────────────────────────


def search(query: str) -> str:
    """Call search_web if available (injected by bridge), else stub."""
    try:
        return search_web(query=query, max_results=3)  # noqa: F821 (bridge-injected)
    except NameError:
        return f"(stub: {query})"


def extract_location(text: str) -> str:
    """Extract a city, state pattern from the first result snippet."""
    # Simple heuristic: look for "City, State" pattern
    m = re.search(r'([A-Z][a-z]+(?: [A-Z][a-z]+)?,\s*[A-Z][a-zA-Z ]+)', text)
    return m.group() if m else text.split("\n")[2][:80] if "\n" in text else text[:80]


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    rows = []
    for entity in ENTITIES:
        query = QUERY_TEMPLATE.format(entity=entity)
        result = search(query)
        value = extract_location(result)
        rows.append({"entity": entity, "value": value, "raw": result.split("\n\n")[0][:200]})
        print(f"  {entity}: {value}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["entity", "value", "raw"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
