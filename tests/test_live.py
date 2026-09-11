import pytest

from agentkit.config import require_env


@pytest.mark.live
def test_anthropic_models_list() -> None:
    import anthropic

    api_key = require_env("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    models = client.models.list()
    assert len(models.data) > 0
