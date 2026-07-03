"""Unit tests for the LLM provider factory."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.graphrag.config import LLMSettings
from src.graphrag.llm import make_llm


def _settings(provider: str = "openai", **overrides) -> LLMSettings:
    base = dict(
        provider=provider,
        openai_api_key="sk-test",
        openai_model="gpt-4o-mini",
        anthropic_api_key="sk-ant-test",
        anthropic_model="claude-sonnet-4-6",
        openrouter_api_key="sk-or-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="anthropic/claude-sonnet-4.6",
    )
    base.update(overrides)
    return LLMSettings(**base)


def test_make_llm_openai():
    with patch("src.graphrag.llm.OpenAILLM") as mock_cls:
        mock_cls.return_value = "fake_llm"
        result = make_llm(_settings("openai"))
    assert result == "fake_llm"
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["model_name"] == "gpt-4o-mini"
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["model_params"] == {"temperature": 0}


def test_make_llm_openai_gpt5_sends_no_temperature():
    """Reasoning models reject a pinned temperature; params must be empty."""
    with patch("src.graphrag.llm.OpenAILLM") as mock_cls:
        mock_cls.return_value = "fake"
        make_llm(_settings("openai", openai_model="gpt-5-mini"))
    assert mock_cls.call_args.kwargs["model_params"] == {}


def test_make_llm_openai_gpt5_point_release_sends_no_temperature():
    with patch("src.graphrag.llm.OpenAILLM") as mock_cls:
        mock_cls.return_value = "fake"
        make_llm(_settings("openai", openai_model="gpt-5.4-mini"))
    assert mock_cls.call_args.kwargs["model_params"] == {}


def test_make_llm_anthropic():
    with patch("src.graphrag.llm.AnthropicLLM") as mock_cls:
        mock_cls.return_value = "fake"
        make_llm(_settings("anthropic"))
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["model_name"] == "claude-sonnet-4-6"
    assert kwargs["api_key"] == "sk-ant-test"
    assert kwargs["model_params"] == {"temperature": 0, "max_tokens": 4096}


def test_make_llm_anthropic_opus_4_8_sends_no_temperature():
    """Opus 4.8 deprecates temperature; only the token budget is sent."""
    with patch("src.graphrag.llm.AnthropicLLM") as mock_cls:
        mock_cls.return_value = "fake"
        make_llm(_settings("anthropic", anthropic_model="claude-opus-4-8"))
    assert mock_cls.call_args.kwargs["model_params"] == {"max_tokens": 4096}


def test_make_llm_openrouter_passes_base_url():
    with patch("src.graphrag.llm.OpenAILLM") as mock_cls:
        mock_cls.return_value = "fake"
        make_llm(_settings("openrouter"))
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert kwargs["api_key"] == "sk-or-test"
    assert kwargs["model_name"] == "anthropic/claude-sonnet-4.6"


def test_make_llm_openrouter_gpt5_sends_no_temperature():
    """The vendor prefix is stripped before the reasoning-model check."""
    with patch("src.graphrag.llm.OpenAILLM") as mock_cls:
        mock_cls.return_value = "fake"
        make_llm(_settings("openrouter", openrouter_model="openai/gpt-5-mini"))
    assert mock_cls.call_args.kwargs["model_params"] == {}


def test_make_llm_openai_missing_key_raises():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_llm(_settings("openai", openai_api_key=""))


def test_make_llm_anthropic_missing_key_raises():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_llm(_settings("anthropic", anthropic_api_key=""))


def test_make_llm_openrouter_missing_key_raises():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_llm(_settings("openrouter", openrouter_api_key=""))


def test_make_llm_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        make_llm(_settings("ollama"))


def test_make_llm_uses_env_when_no_settings_passed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    with patch("src.graphrag.llm.OpenAILLM") as mock_cls:
        mock_cls.return_value = "fake"
        make_llm()
    assert mock_cls.call_args.kwargs["api_key"] == "from-env"


class _Obj:
    """Attribute bag for faking SDK response shapes."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_openai_response(reasoning: int | None = 30) -> _Obj:
    return _Obj(
        choices=[_Obj(message=_Obj(content="hi"))],
        usage=_Obj(
            prompt_tokens=100,
            completion_tokens=50,
            completion_tokens_details=_Obj(reasoning_tokens=reasoning),
            prompt_tokens_details=_Obj(cached_tokens=10),
        ),
    )


def _fake_anthropic_response() -> _Obj:
    return _Obj(
        content=[_Obj(type="text", text="hello")],
        usage=_Obj(input_tokens=80, output_tokens=20, cache_read_input_tokens=5),
    )


class TestUsageTrackingLLM:
    def test_openai_invoke_surfaces_provider_usage(self):
        from src.graphrag.llm import UsageTrackingLLM

        with patch("openai.OpenAI") as mock_cls:
            client = mock_cls.return_value
            client.chat.completions.create.return_value = _fake_openai_response()
            llm = UsageTrackingLLM("openai", "gpt-5-mini", "sk-test")
            resp = llm.invoke("prompt")
        assert resp.content == "hi"
        assert resp.usage_metadata == {
            "input_tokens": 100,
            "output_tokens": 50,
            "reasoning_tokens": 30,
            "cached_input_tokens": 10,
        }
        # Reasoning model: no pinned temperature in the request.
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "temperature" not in kwargs

    def test_openai_non_reasoning_model_pins_temperature(self):
        from src.graphrag.llm import UsageTrackingLLM

        with patch("openai.OpenAI") as mock_cls:
            client = mock_cls.return_value
            client.chat.completions.create.return_value = _fake_openai_response(
                reasoning=None
            )
            llm = UsageTrackingLLM("openai", "gpt-4o-mini", "sk-test")
            resp = llm.invoke("prompt")
        assert resp.usage_metadata["reasoning_tokens"] == 0
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0

    def test_anthropic_invoke_surfaces_provider_usage(self):
        from src.graphrag.llm import UsageTrackingLLM

        with patch("anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            client.messages.create.return_value = _fake_anthropic_response()
            llm = UsageTrackingLLM("anthropic", "claude-opus-4-8", "sk-ant")
            resp = llm.invoke("prompt")
        assert resp.content == "hello"
        assert resp.usage_metadata == {
            "input_tokens": 80,
            "output_tokens": 20,
            "reasoning_tokens": 0,
            "cached_input_tokens": 5,
        }
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == 4096
        assert "temperature" not in kwargs  # opus 4.8 deprecates it

    def test_make_usage_llm_routes_providers(self):
        from src.graphrag.llm import make_usage_llm

        with patch("openai.OpenAI"), patch("anthropic.Anthropic"):
            assert make_usage_llm(_settings("openai")).provider == "openai"
            assert make_usage_llm(_settings("anthropic")).provider == "anthropic"
            router = make_usage_llm(_settings("openrouter"))
            assert router.provider == "openrouter"
            assert router.model_name == "anthropic/claude-sonnet-4.6"

    def test_make_usage_llm_missing_key_raises(self):
        from src.graphrag.llm import make_usage_llm

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            make_usage_llm(_settings("openai", openai_api_key=""))

    def test_make_usage_llm_anthropic_missing_key_raises(self):
        from src.graphrag.llm import make_usage_llm

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            make_usage_llm(_settings("anthropic", anthropic_api_key=""))

    def test_make_usage_llm_openrouter_missing_key_raises(self):
        from src.graphrag.llm import make_usage_llm

        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            make_usage_llm(_settings("openrouter", openrouter_api_key=""))

    def test_make_usage_llm_unknown_provider_raises(self):
        from src.graphrag.llm import make_usage_llm

        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            make_usage_llm(_settings("ollama"))
