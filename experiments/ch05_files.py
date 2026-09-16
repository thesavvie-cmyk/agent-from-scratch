"""Chapter 5 file tools & callback experiments (block 11).

Sections
--------
a) Structural search: test project, agent finds which database is used
b) Security: Workspace blocks ../../.env and /etc/passwd
c) read_excel and analyze_image on generated test files
d) Approval callback: delete_file with confirm and deny
e) Search compressor: same question with/without callback, token comparison

Usage
-----
    uv run python experiments/ch05_files.py --section a
    uv run python experiments/ch05_files.py --section all

Requires ANTHROPIC_API_KEY for sections a, c, d, e.
"""
from __future__ import annotations

import argparse
import asyncio
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

# ── helpers ────────────────────────────────────────────────────────────────────


def _hr(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def _build_test_project(root: Path) -> None:
    """Create a minimal Python project in *root* for section (a)."""
    root.mkdir(parents=True, exist_ok=True)

    (root / "README.md").write_text(
        "# My Project\n\nA sample application.\n"
    )
    (root / "requirements.txt").write_text(
        "flask==3.0.0\npsycopg2-binary==2.9.9\nsqlalchemy==2.0.0\n"
    )

    src = root / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "database.py").write_text(
        '"""Database connection module."""\n'
        "import psycopg2\n\n"
        "DATABASE_URL = \"postgresql://user:pass@localhost:5432/mydb\"\n\n"
        "def get_connection():\n"
        '    return psycopg2.connect(DATABASE_URL)\n'
    )
    (src / "app.py").write_text(
        "from flask import Flask\nfrom src.database import get_connection\n\n"
        "app = Flask(__name__)\n"
    )

    cfg = root / "config"
    cfg.mkdir()
    (cfg / "settings.json").write_text(
        '{\n  "db_host": "localhost",\n  "db_port": 5432,\n'
        '  "db_name": "mydb",\n  "db_driver": "postgresql"\n}\n'
    )


def _make_png(path: Path) -> None:
    """Create a 6x4 PNG with three colored horizontal stripes (RGB, no PIL)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return length + tag + data + crc

    width, height = 6, 6
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    # Three 2-pixel-tall colored bands: red, green, blue
    raw = b""
    palette = [(220, 50, 50), (50, 200, 50), (50, 50, 220)]
    for color in palette:
        for _ in range(2):  # 2 rows per color
            row = bytes([0])  # filter byte = None
            for _ in range(width):
                row += bytes(color)
            raw += row

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _make_excel(path: Path) -> None:
    """Create a simple .xlsx with project dependency data."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Dependencies"
    ws.append(["Package", "Version", "Purpose"])
    ws.append(["flask", "3.0.0", "Web framework"])
    ws.append(["psycopg2-binary", "2.9.9", "PostgreSQL driver"])
    ws.append(["sqlalchemy", "2.0.0", "ORM"])
    ws.append(["pydantic", "2.5.0", "Data validation"])
    wb.save(path)


def _make_agent(tools: list[Any], callbacks: Any = None) -> Any:
    from agentkit.agent import Agent
    from agentkit.config import FAST_MODEL
    from agentkit.llm import LlmClient

    return Agent(
        model=LlmClient(FAST_MODEL),
        tools=tools,
        instructions="You are a helpful coding assistant. Use tools to read files.",
        max_steps=8,
        callbacks=callbacks,
    )


def _print_trace(result: Any) -> None:
    from agentkit.context import ExecutionContext
    from agentkit.types import Message, ToolCall, ToolResult

    ctx: ExecutionContext = result.context
    for event in ctx.events:
        for item in event.content:
            if isinstance(item, Message) and item.role == "user":
                print(f"\n[USER] {item.content[:120]}")
            elif isinstance(item, Message):
                print(f"\n[ASSISTANT] {item.content[:300]}")
            elif isinstance(item, ToolCall):
                args_str = str(item.arguments)[:80]
                print(f"\n[TOOL CALL] {item.name}({args_str})")
            elif isinstance(item, ToolResult):
                content_str = str(item.content[0])[:200] if item.content else ""
                print(f"[TOOL RESULT/{item.status}] {content_str}")


# ── Section A — structural search ─────────────────────────────────────────────


async def section_a() -> None:
    _hr("Section A — Structural search: which database does the project use?")
    from agentkit.context import ExecutionContext
    from agentkit.tools.files import Workspace, list_files, read_file, search_in_files

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "myproject"
        _build_test_project(root)

        ctx = ExecutionContext()
        ctx.state["workspace"] = Workspace(root)

        agent = _make_agent([list_files, read_file, search_in_files])
        result = await agent.run(
            "What database does this project use? Look at the project files.",
            context=ctx,
        )

        _print_trace(result)
        print(f"\n[FINAL ANSWER] {result.output}")

        # Summary: which files were opened
        from agentkit.types import ToolCall
        opened = [
            item.arguments.get("path", "?")
            for event in ctx.events
            for item in event.content
            if isinstance(item, ToolCall) and item.name in ("read_file", "search_in_files")
        ]
        print(f"\n[FILES ACCESSED] {opened}")


# ── Section B — security ───────────────────────────────────────────────────────


async def section_b() -> None:
    _hr("Section B — Workspace blocks path traversal")
    from agentkit.context import ExecutionContext
    from agentkit.tools.files import Workspace, read_file

    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Workspace(tmpdir)
        ctx = ExecutionContext()
        ctx.state["workspace"] = ws

        for bad_path in ["../../.env", "/etc/passwd", "../../../Windows/System32"]:
            result = await read_file.execute(ctx, path=bad_path)
            blocked = "Error" in result
            print(f"  {'BLOCKED' if blocked else 'ALLOWED (BUG!)'}: {bad_path!r}")
            print(f"  → {result[:120]}")


# ── Section C — Excel and image ────────────────────────────────────────────────


async def section_c() -> None:
    _hr("Section C — read_excel and analyze_image")
    from agentkit.context import ExecutionContext
    from agentkit.tools.documents import analyze_image, read_excel
    from agentkit.tools.files import Workspace

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        xlsx = root / "deps.xlsx"
        png = root / "stripes.png"
        _make_excel(xlsx)
        _make_png(png)

        ctx = ExecutionContext()
        ctx.state["workspace"] = Workspace(root)

        print("\n  Excel:")
        excel_result = await read_excel.execute(ctx, path="deps.xlsx")
        print(excel_result)

        print("\n  Image (analyze_image):")
        img_result = await analyze_image.execute(
            ctx, path="stripes.png", query="Describe what you see in this image."
        )
        print(img_result[:400])


# ── Section D — Approval callback ─────────────────────────────────────────────


async def section_d() -> None:
    _hr("Section D — Approval callback: delete_file")
    from agentkit.callbacks import Callbacks
    from agentkit.callbacks_builtin import make_approval_callback
    from agentkit.context import ExecutionContext
    from agentkit.tools.base import tool
    from agentkit.tools.files import Workspace

    @tool
    def delete_file(context: Any, path: str) -> str:
        """Delete a file from the workspace. DANGEROUS — cannot be undone."""
        from agentkit.tools.files import _get_workspace
        ws = _get_workspace(context)
        p = ws.resolve(path)
        if not p.exists():
            return f"Error: {path!r} not found"
        p.unlink()
        return f"Deleted: {path}"

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "important.txt").write_text("Do not delete me!")
        (root / "temp.txt").write_text("Temporary file")

        print("\n  Scenario 1 — User ALLOWS deletion (simulated 'y'):")
        allow_cb = make_approval_callback({"delete_file"}, prompt_fn=lambda _: "y")
        agent = _make_agent([delete_file], callbacks=Callbacks(before_tool=[allow_cb]))
        ctx = ExecutionContext()
        ctx.state["workspace"] = Workspace(root)
        result = await agent.run("Delete the file temp.txt", context=ctx)
        print(f"  Agent: {result.output[:200]}")
        print(f"  temp.txt exists: {(root / 'temp.txt').exists()}")

        print("\n  Scenario 2 — User DENIES deletion (simulated 'n'):")
        deny_cb = make_approval_callback({"delete_file"}, prompt_fn=lambda _: "n")
        agent = _make_agent([delete_file], callbacks=Callbacks(before_tool=[deny_cb]))
        ctx2 = ExecutionContext()
        ctx2.state["workspace"] = Workspace(root)
        result2 = await agent.run("Delete the file important.txt", context=ctx2)
        print(f"  Agent: {result2.output[:200]}")
        print(f"  important.txt exists: {(root / 'important.txt').exists()}")


# ── Section E — Search compressor comparison ──────────────────────────────────


async def section_e() -> None:
    _hr("Section E — Search compressor: token comparison")
    from agentkit.callbacks import Callbacks
    from agentkit.callbacks_builtin import make_search_compressor
    from agentkit.context import ExecutionContext
    from agentkit.retrieval import VectorIndex
    from agentkit.tools.base import tool
    from agentkit.tools.files import Workspace

    try:
        import importlib.util
        if importlib.util.find_spec("fastembed") is None:
            raise ImportError
        from agentkit.embeddings import LocalEmbeddings
        def _index_factory() -> Any:
            return VectorIndex(LocalEmbeddings())
    except ImportError:
        print("  [SKIP] fastembed not available (uv sync --group local)")
        return

    # Build a "long search result" mock tool
    LONG_RESULT = (
        "PostgreSQL is an advanced open-source relational database. "
        "It supports ACID transactions, complex queries, foreign keys, triggers, "
        "views, and stored procedures. PostgreSQL uses a client-server model. "
        "The server process manages database files, handles connections from clients, "
        "and performs database actions on behalf of the clients. "
        "SQLite is a serverless, self-contained database engine widely used for "
        "embedded databases and mobile applications. It stores the entire database "
        "in a single cross-platform file. SQLite supports most of the SQL standard. "
        "MySQL is a widely-used open-source relational database management system. "
        "It is especially popular for web applications and is a component of the "
        "LAMP stack. MySQL supports replication, clustering, and partitioning. "
        "MongoDB is a document-oriented NoSQL database. It stores data in flexible "
        "JSON-like documents. MongoDB is designed for scalability and developer agility. "
    ) * 6  # ~1100 tokens

    @tool
    def search_docs(context: Any, query: str) -> str:
        """Search project documentation."""
        return LONG_RESULT

    question = "What database does the project use and why?"

    rows = []
    for label, use_compressor in [("without compressor", False), ("with compressor", True)]:
        cb = None
        if use_compressor:
            compressor = make_search_compressor(_index_factory, top_k=3, min_tokens=100)
            cb = Callbacks(after_tool=[compressor])

        agent = _make_agent([search_docs], callbacks=cb)
        ctx = ExecutionContext()
        ctx.state["workspace"] = Workspace(tempfile.mkdtemp())

        result = await agent.run(question, context=ctx)
        usage = ctx.state.get("token_usage", {})
        total_in = usage.get("input_tokens", 0)
        total_out = usage.get("output_tokens", 0)
        steps = ctx.current_step
        rows.append((label, total_in, total_out, steps, result.output[:80]))

    # Print table
    print(f"\n  {'Config':<25} {'Input tok':>10} {'Output tok':>11} {'Steps':>6}")
    print(f"  {'─' * 25} {'─' * 10} {'─' * 11} {'─' * 6}")
    for label, inp, out, steps, _ in rows:
        print(f"  {label:<25} {inp:>10} {out:>11} {steps:>6}")

    print("\n  Answers:")
    for label, _, _, _, ans in rows:
        print(f"  [{label}] {ans}")


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Block 11 file tool experiments")
    parser.add_argument(
        "--section",
        default="all",
        choices=["a", "b", "c", "d", "e", "all"],
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    sections = {
        "a": section_a,
        "b": section_b,
        "c": section_c,
        "d": section_d,
        "e": section_e,
    }
    to_run = list(sections) if args.section == "all" else [args.section]
    for key in to_run:
        try:
            asyncio.run(sections[key]())
        except KeyboardInterrupt:
            print("\n[interrupted]")
            sys.exit(1)
        except Exception:  # noqa: BLE001
            import traceback
            print(f"\n[ERROR in section {key}]")
            print(traceback.format_exc())

    print()


if __name__ == "__main__":
    main()
