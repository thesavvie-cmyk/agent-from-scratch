"""File system tools with Workspace security model (block 11).

Security model
--------------
Workspace.resolve() rejects any path that escapes the root after resolving
symlinks and '..'. This prevents directory traversal: an agent on a server
with a public IP cannot read ~/.ssh, .env, or /etc/passwd through these tools.

Workspace is stored in context.state["workspace"]. Tools raise RuntimeError
if no workspace is configured — silent fallback to CWD would be unsafe.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from agentkit.context import ExecutionContext
from agentkit.tools.base import tool

_READ_LIMIT_BYTES = 51_200   # 50 KB
_SEARCH_MAX_MATCHES = 200


# ── Workspace ──────────────────────────────────────────────────────────────────


class WorkspaceEscapeError(PermissionError):
    """Raised when a path resolves outside the workspace root."""


class Workspace:
    """A bounded filesystem sandbox.

    Every path passed to file tools is resolved through this class. Symlinks
    are followed; if the resolved path is outside ``root``, WorkspaceEscapeError
    is raised before any I/O occurs.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, path: str | Path) -> Path:
        """Resolve *path* within the workspace.

        Absolute paths are checked directly. Relative paths are joined to the
        root first. After resolving symlinks and normalising '..', raises
        WorkspaceEscapeError if the result is outside the root.
        """
        p = Path(path)
        candidate = p.resolve() if p.is_absolute() else (self._root / p).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError:
            raise WorkspaceEscapeError(
                f"Path {str(path)!r} resolves to {candidate}, "
                f"which is outside workspace root {self._root}"
            )
        return candidate


def _get_workspace(context: ExecutionContext) -> Workspace:
    ws = context.state.get("workspace")
    if ws is None:
        raise RuntimeError(
            "No workspace set. Do context.state['workspace'] = Workspace(path) "
            "before using file tools."
        )
    return ws


# ── Helpers ────────────────────────────────────────────────────────────────────


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n //= 1024
    return f"{n}GB"


# ── Tools ──────────────────────────────────────────────────────────────────────


@tool
def list_files(
    context: ExecutionContext,
    path: str = ".",
    max_depth: int = 3,
) -> str:
    """List files and directories as a tree with sizes.

    path: directory relative to workspace root (default: root)
    max_depth: maximum recursion depth (default: 3)
    """
    ws = _get_workspace(context)
    root = ws.resolve(path)
    if not root.is_dir():
        return f"Error: {path!r} is not a directory"

    lines: list[str] = [str(root)]

    def _walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda e: (e.is_file(), e.name.lower()),
            )
        except PermissionError:
            lines.append(f"{prefix}[permission denied]")
            return
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            if entry.is_symlink():
                try:
                    target = os.readlink(entry)
                except OSError:
                    target = "?"
                lines.append(f"{prefix}{connector}{entry.name} -> {target} [symlink]")
            elif entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                _walk(entry, child_prefix, depth + 1)
            else:
                try:
                    size = _human_size(entry.stat().st_size)
                except OSError:
                    size = "?"
                lines.append(f"{prefix}{connector}{entry.name} ({size})")

    _walk(root, "", 1)
    return "\n".join(lines)


@tool
def read_file(
    context: ExecutionContext,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read a text file from the workspace, optionally a line range.

    path: file path relative to workspace root
    start_line: first line to return (1-based inclusive; default: 1)
    end_line: last line to return (1-based inclusive; default: end)

    Returns at most 50 KB. If the file is larger, the output is truncated
    with a message showing how many lines remain.
    """
    ws = _get_workspace(context)
    try:
        p = ws.resolve(path)
    except WorkspaceEscapeError as exc:
        return f"Error: {exc}"
    if not p.exists():
        return f"Error: file not found: {path!r}"
    if not p.is_file():
        return f"Error: {path!r} is not a regular file"

    try:
        raw = p.read_bytes()
    except OSError as exc:
        return f"Error reading {path!r}: {exc}"

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError:
            return f"Error: {path!r} does not appear to be a text file"

    lines = text.splitlines(keepends=True)
    total = len(lines)
    lo = max(1, start_line or 1)
    hi = min(total, end_line if end_line is not None else total)
    selected = lines[lo - 1 : hi]

    parts: list[str] = []
    running = 0
    truncated_at: int | None = None
    for i, line in enumerate(selected, lo):
        encoded = line.encode("utf-8")
        if running + len(encoded) > _READ_LIMIT_BYTES:
            truncated_at = i
            break
        parts.append(line)
        running += len(encoded)

    output = "".join(parts)
    if truncated_at is not None:
        output += (
            f"\n[Truncated at line {truncated_at}: output limit reached. "
            f"File has {total} lines total. Use start_line/end_line to read more.]"
        )
    elif start_line is not None or end_line is not None:
        shown_hi = min(hi, lo + len(parts) - 1)
        output += f"\n[Lines {lo}–{shown_hi} of {total}]"

    return output


@tool
def search_in_files(
    context: ExecutionContext,
    pattern: str,
    path: str = ".",
    is_regex: bool = False,
) -> str:
    """Search for a pattern across files in the workspace.

    pattern: literal string or regex to search for
    path: directory or file (relative to workspace root; default: root)
    is_regex: treat pattern as Python regex (default: False)

    Returns matches in grep format: path:lineno:content.
    Limited to 200 matches.
    """
    ws = _get_workspace(context)
    try:
        target = ws.resolve(path)
    except WorkspaceEscapeError as exc:
        return f"Error: {exc}"

    if is_regex:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex {pattern!r}: {exc}"
        def _matches(line: str) -> bool:
            return bool(compiled.search(line))
    else:
        def _matches(line: str) -> bool:  # type: ignore[misc]
            return pattern in line

    files: list[Path]
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(f for f in target.rglob("*") if f.is_file())
    else:
        return f"Error: {path!r} is not a file or directory"

    results: list[str] = []
    for filepath in files:
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            rel = filepath.relative_to(ws.root)
        except ValueError:
            rel = filepath
        for lineno, line in enumerate(text.splitlines(), 1):
            if _matches(line):
                results.append(f"{rel}:{lineno}:{line.rstrip()}")
                if len(results) >= _SEARCH_MAX_MATCHES:
                    results.append(f"[Truncated at {_SEARCH_MAX_MATCHES} matches]")
                    return "\n".join(results)

    if not results:
        return f"No matches found for {pattern!r} in {path!r}"
    return "\n".join(results)
