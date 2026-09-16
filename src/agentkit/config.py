from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

FAST_MODEL: str = os.getenv("FAST_MODEL", "anthropic/claude-haiku-4-5")
SMART_MODEL: str = os.getenv("SMART_MODEL", "anthropic/claude-sonnet-5")

TRACKED_KEYS: list[str] = [
    "ANTHROPIC_API_KEY",
    "TAVILY_API_KEY",
    "HF_TOKEN",
    "E2B_API_KEY",
]


class MissingEnvError(RuntimeError):
    def __init__(self, name: str) -> None:
        super().__init__(
            f"Environment variable '{name}' is not set. Add it to .env"
        )
        self.name = name


def require_env(name: str) -> str:
    """Return the value of an environment variable or raise MissingEnvError."""
    value = os.getenv(name)
    if not value:
        raise MissingEnvError(name)
    return value


def find_uv() -> str:
    """Return the path to the uv binary.

    Checks PATH first, then the common user-local install location
    (~/.local/bin/uv on Linux/macOS).  Falls back to bare "uv" so the
    OS error message stays informative when it's truly missing.
    """
    found = shutil.which("uv")
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "uv"
    if candidate.exists():
        return str(candidate)
    return "uv"


def env_status() -> dict[str, bool]:
    """Return which tracked API keys are currently set."""
    return {key: bool(os.getenv(key)) for key in TRACKED_KEYS}
