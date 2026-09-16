"""Tests for bench --compare (block 9)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_bench_result(hostname: str, elapsed: float, steps: float) -> dict:
    questions = [
        {
            "id": f"q{i}",
            "question": "...",
            "needs_tools": i > 2,
            "elapsed_s": elapsed + i * 0.1,
            "steps": int(steps),
            "input_tokens": 100 * i,
            "output_tokens": 20 * i,
            "memory_delta_mb": 1.0,
            "error": None,
            "output": "answer",
        }
        for i in range(1, 6)
    ]
    return {
        "hostname": hostname,
        "model": "anthropic/claude-haiku-4-5",
        "mcp_cold_start_s": 1.5 if hostname == "server" else 0.5,
        "questions": questions,
        "summary": {
            "n": 5,
            "total_elapsed_s": sum(q["elapsed_s"] for q in questions),
            "avg_elapsed_s": elapsed + 0.3,
            "avg_steps": steps,
            "total_input_tokens": sum(q["input_tokens"] for q in questions),
            "total_output_tokens": sum(q["output_tokens"] for q in questions),
            "avg_memory_delta_mb": 1.0,
            "errors": 0,
        },
    }


def test_compare_prints_table(tmp_path: Path, capsys) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from bench import _compare

    file_a = tmp_path / "bench-pc.json"
    file_b = tmp_path / "bench-server.json"
    file_a.write_text(json.dumps(_make_bench_result("pc", 1.0, 2)), encoding="utf-8")
    file_b.write_text(json.dumps(_make_bench_result("server", 3.0, 2)), encoding="utf-8")

    _compare(str(file_a), str(file_b))

    out = capsys.readouterr().out
    assert "q1" in out
    assert "pc" in out
    assert "server" in out
    assert "avg_elapsed_s" in out


def test_compare_delta_positive_when_server_slower(tmp_path: Path, capsys) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from bench import _compare

    file_a = tmp_path / "a.json"
    file_b = tmp_path / "b.json"
    file_a.write_text(json.dumps(_make_bench_result("fast", 1.0, 2)), encoding="utf-8")
    file_b.write_text(json.dumps(_make_bench_result("slow", 5.0, 2)), encoding="utf-8")

    _compare(str(file_a), str(file_b))

    out = capsys.readouterr().out
    # All deltas should be positive (b is slower than a)
    lines = [l for l in out.splitlines() if "q" in l and "|" not in l and "=" not in l]
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            delta_str = parts[3]
            if delta_str.startswith("+"):
                assert float(delta_str) > 0


def test_summarise_function() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from bench import _summarise

    rows = [
        {"elapsed_s": 1.0, "steps": 2, "input_tokens": 100, "output_tokens": 20, "memory_delta_mb": 5.0, "error": None},
        {"elapsed_s": 2.0, "steps": 3, "input_tokens": 200, "output_tokens": 40, "memory_delta_mb": 3.0, "error": "oops"},
    ]
    s = _summarise(rows)
    assert s["n"] == 2
    assert s["errors"] == 1
    assert s["avg_elapsed_s"] == pytest.approx(1.5)
    assert s["total_input_tokens"] == 300
