import pytest

from agentkit.config import MissingEnvError, env_status, require_env


def test_require_env_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_VAR_X", "hello")
    assert require_env("TEST_VAR_X") == "hello"


def test_require_env_raises_on_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_VAR_X", raising=False)
    with pytest.raises(MissingEnvError) as exc_info:
        require_env("MISSING_VAR_X")
    assert "MISSING_VAR_X" in str(exc_info.value)


def test_env_status_reflects_set_and_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("E2B_API_KEY", raising=False)

    status = env_status()
    assert status["ANTHROPIC_API_KEY"] is True
    assert status["TAVILY_API_KEY"] is False
    assert status["HF_TOKEN"] is False
    assert status["E2B_API_KEY"] is False
