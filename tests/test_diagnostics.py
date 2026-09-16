"""Tests for diagnostics / doctor command (block 9)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

# ── Individual check functions ────────────────────────────────────────────────


def test_check_platform_returns_dict() -> None:
    from agentkit.diagnostics import _check_platform

    result = _check_platform()
    assert "platform" in result
    assert "python" in result
    assert "arch" in result
    assert isinstance(result["platform"], str)


def test_check_disk_returns_positive_values() -> None:
    from agentkit.diagnostics import _check_disk

    result = _check_disk()
    assert result["disk_free_gb"] >= 0
    assert result["disk_total_gb"] > 0


def test_check_memory_no_crash_if_psutil_missing() -> None:
    """_check_memory must not raise even when psutil is not installed."""
    from agentkit.diagnostics import _check_memory

    with patch.dict("sys.modules", {"psutil": None}):
        result = _check_memory()
    assert "ram_free_gb" in result
    assert "ram_total_gb" in result


def test_check_memory_with_psutil() -> None:
    from agentkit.diagnostics import _check_memory

    result = _check_memory()
    # If psutil is installed in test env, values should be numeric
    if result["ram_free_gb"] != "N/A":
        assert isinstance(result["ram_free_gb"], float)
        assert result["ram_free_gb"] >= 0


def test_check_node_missing() -> None:
    """_check_node returns NOT FOUND when node/npx are absent."""
    from agentkit.diagnostics import _check_node

    with patch("agentkit.diagnostics._run_version", return_value=None):
        result = _check_node()
    assert result["node"] == "NOT FOUND"
    assert result["npx"] == "NOT FOUND"
    assert result["node_ok"] is False
    assert result["npx_ok"] is False


def test_check_node_present() -> None:
    from agentkit.diagnostics import _check_node

    with patch("agentkit.diagnostics._run_version", return_value="v20.0.0"):
        result = _check_node()
    assert result["node_ok"] is True
    assert "v20" in result["node"]


def test_check_env_keys_returns_dict() -> None:
    from agentkit.diagnostics import _check_env_keys

    result = _check_env_keys()
    assert isinstance(result, dict)
    assert "ANTHROPIC_API_KEY" in result


def test_check_anthropic_api_no_key() -> None:
    from agentkit.diagnostics import _check_anthropic_api

    with patch.dict("os.environ", {}, clear=True):
        result = _check_anthropic_api()
    assert result["anthropic_ok"] is False
    assert "not set" in result["anthropic_error"]


def test_check_anthropic_api_bad_key() -> None:
    from agentkit.diagnostics import _check_anthropic_api

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "bad-key"}),
        patch("anthropic.Anthropic") as mock_cls,
    ):
        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("401 Unauthorized")
        mock_cls.return_value = mock_client
        result = _check_anthropic_api()
    assert result["anthropic_ok"] is False
    assert "anthropic_error" in result


def test_check_anthropic_api_success() -> None:
    from agentkit.diagnostics import _check_anthropic_api

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-valid"}),
        patch("anthropic.Anthropic") as mock_cls,
    ):
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_cls.return_value = mock_client
        result = _check_anthropic_api()
    assert result["anthropic_ok"] is True


# ── run_doctor ────────────────────────────────────────────────────────────────


async def test_run_doctor_no_crash_without_api_key(capsys) -> None:
    """doctor must complete (not raise) even when keys are missing."""
    from agentkit.diagnostics import run_doctor

    with (
        patch("agentkit.diagnostics._check_anthropic_api", return_value={"anthropic_ok": False, "anthropic_error": "no key"}),
        patch("agentkit.diagnostics._check_mcp_cold_start", new=AsyncMock(return_value={"mcp_ok": False, "mcp_cold_start_s": 0.1})),
    ):
        exit_code = await run_doctor(json_output=False)

    assert exit_code == 1  # critical check failed


async def test_run_doctor_json_output(capsys) -> None:
    """--json flag produces valid JSON with expected top-level keys."""
    from agentkit.diagnostics import run_doctor

    with (
        patch("agentkit.diagnostics._check_anthropic_api", return_value={"anthropic_ok": True}),
        patch("agentkit.diagnostics._check_mcp_cold_start", new=AsyncMock(return_value={"mcp_ok": True, "mcp_cold_start_s": 0.5})),
    ):
        exit_code = await run_doctor(json_output=True)

    out = capsys.readouterr().out
    # The output contains "Checking ..." lines before the JSON block; find the
    # first line that starts '{' and parse from there.
    json_start = out.find("{")
    assert json_start >= 0, "Expected JSON in output"
    data = json.loads(out[json_start:])
    assert "system" in data
    assert "anthropic" in data
    assert "mcp" in data
    assert exit_code == 0


async def test_run_doctor_returns_zero_on_success(capsys) -> None:
    from agentkit.diagnostics import run_doctor

    with (
        patch("agentkit.diagnostics._check_anthropic_api", return_value={"anthropic_ok": True}),
        patch("agentkit.diagnostics._check_mcp_cold_start", new=AsyncMock(return_value={"mcp_ok": True, "mcp_cold_start_s": 0.3})),
        patch("agentkit.config.env_status", return_value={"ANTHROPIC_API_KEY": True, "TAVILY_API_KEY": True, "HF_TOKEN": False, "E2B_API_KEY": False}),
    ):
        exit_code = await run_doctor(json_output=False)

    assert exit_code == 0
