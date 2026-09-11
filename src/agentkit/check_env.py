from __future__ import annotations

import os

from agentkit.config import TRACKED_KEYS


def _mask(value: str) -> str:
    return value[:4] + "***" if len(value) >= 4 else "***"


def main() -> None:
    print("=== Environment Status ===")
    for key in TRACKED_KEYS:
        value = os.getenv(key, "")
        if value:
            print(f"  {key}: SET ({_mask(value)})")
        else:
            print(f"  {key}: NOT SET")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        models = client.models.list()
        ids = [m.id for m in models.data]
        has_haiku = any("haiku" in m for m in ids)
        has_sonnet = any("sonnet" in m for m in ids)
        print()
        print("=== Anthropic API ===")
        print(f"  Models available: {len(ids)}")
        print(f"  haiku:  {'yes' if has_haiku else 'no'}")
        print(f"  sonnet: {'yes' if has_sonnet else 'no'}")
    except anthropic.APIError as exc:
        print(f"\n  Anthropic API error: {exc}")


if __name__ == "__main__":
    main()
