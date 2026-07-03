"""LLM factory."""

from __future__ import annotations

from neo4j_graphrag.llm import AnthropicLLM, LLMInterface, OpenAILLM

from .config import LLMSettings

# Anthropic requires max_tokens on every call; the one-shot weekly plan
# JSON runs past 1024 output tokens for a 20-task week, which truncated
# the array mid-placement.
ANTHROPIC_MAX_TOKENS = 4096


def _openai_model_params(model_name: str) -> dict:
    """Reasoning models accept only the default temperature; send no pin."""
    base = model_name.rsplit("/", 1)[-1]
    if base.startswith(("gpt-5", "o1", "o3", "o4")):
        return {}
    return {"temperature": 0}


def _anthropic_model_params(model_name: str) -> dict:
    """Opus 4.8 deprecates temperature; max_tokens stays mandatory."""
    if model_name.startswith("claude-opus-4-8"):
        return {"max_tokens": ANTHROPIC_MAX_TOKENS}
    return {"temperature": 0, "max_tokens": ANTHROPIC_MAX_TOKENS}


def make_llm(settings: LLMSettings | None = None) -> LLMInterface:
    s = settings or LLMSettings.from_env()
    provider = s.provider

    if provider == "openai":
        if not s.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty")
        return OpenAILLM(
            model_name=s.openai_model,
            api_key=s.openai_api_key,
            model_params=_openai_model_params(s.openai_model),
        )

    if provider == "anthropic":
        if not s.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty")
        return AnthropicLLM(
            model_name=s.anthropic_model,
            api_key=s.anthropic_api_key,
            model_params=_anthropic_model_params(s.anthropic_model),
        )

    if provider == "openrouter":
        if not s.openrouter_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is empty"
            )
        return OpenAILLM(
            model_name=s.openrouter_model,
            api_key=s.openrouter_api_key,
            base_url=s.openrouter_base_url,
            model_params=_openai_model_params(s.openrouter_model),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. Use one of: openai, anthropic, openrouter."
    )


class UsageResponse:
    """Chat response carrying the provider-reported token usage.

    `usage_metadata` keys: `input_tokens`, `output_tokens`,
    `reasoning_tokens` (OpenAI reasoning models; 0 elsewhere), and
    `cached_input_tokens`. Values are the provider's own counts, not
    tokenizer estimates.
    """

    def __init__(self, content: str, usage_metadata: dict) -> None:
        self.content = content
        self.usage_metadata = usage_metadata


class UsageTrackingLLM:
    """Chat client over the raw provider SDKs that keeps usage metadata.

    Exposes the same `invoke(prompt) -> response-with-.content` shape as
    the `neo4j_graphrag` clients so it drops into `DirectLLMPipeline`,
    the judge oracle, and the paraphrase stage unchanged, while the
    response additionally carries `usage_metadata` for telemetry.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model
        if provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
        else:
            import openai

            self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def invoke(self, prompt: str) -> UsageResponse:
        if self.provider == "anthropic":
            return self._invoke_anthropic(prompt)
        return self._invoke_openai(prompt)

    def _invoke_openai(self, prompt: str) -> UsageResponse:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **_openai_model_params(self.model_name),
        )
        content = (
            (response.choices[0].message.content or "") if response.choices else ""
        )
        usage = getattr(response, "usage", None)
        out_details = getattr(usage, "completion_tokens_details", None)
        in_details = getattr(usage, "prompt_tokens_details", None)
        return UsageResponse(
            content,
            {
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "reasoning_tokens": int(
                    getattr(out_details, "reasoning_tokens", 0) or 0
                ),
                "cached_input_tokens": int(
                    getattr(in_details, "cached_tokens", 0) or 0
                ),
            },
        )

    def _invoke_anthropic(self, prompt: str) -> UsageResponse:
        response = self._client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **_anthropic_model_params(self.model_name),
        )
        content = "".join(
            block.text
            for block in (response.content or [])
            if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        # Anthropic bills thinking as output tokens but does not break
        # them out in `usage`; reasoning stays 0 unless thinking blocks
        # are explicitly enabled and counted upstream.
        return UsageResponse(
            content,
            {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "reasoning_tokens": 0,
                "cached_input_tokens": int(
                    getattr(usage, "cache_read_input_tokens", 0) or 0
                ),
            },
        )


def make_usage_llm(settings: LLMSettings | None = None) -> UsageTrackingLLM:
    """Build a `UsageTrackingLLM` for the direct chat paths.

    Same provider/model/key resolution as `make_llm`; use this wherever
    the call site only needs `invoke()` (augmenter, judge oracle,
    paraphrase) so telemetry gets provider-reported token counts. The
    GraphRAG retrieval pipeline keeps the `neo4j_graphrag` client from
    `make_llm`.
    """
    s = settings or LLMSettings.from_env()
    provider = s.provider

    if provider == "openai":
        if not s.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty")
        return UsageTrackingLLM("openai", s.openai_model, s.openai_api_key)

    if provider == "anthropic":
        if not s.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty")
        return UsageTrackingLLM("anthropic", s.anthropic_model, s.anthropic_api_key)

    if provider == "openrouter":
        if not s.openrouter_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is empty"
            )
        return UsageTrackingLLM(
            "openrouter",
            s.openrouter_model,
            s.openrouter_api_key,
            base_url=s.openrouter_base_url,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. Use one of: openai, anthropic, openrouter."
    )
