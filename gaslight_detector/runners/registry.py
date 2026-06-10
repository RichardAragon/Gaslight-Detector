from __future__ import annotations

from typing import Any

_PROVIDERS = {"openai", "anthropic", "local", "replay"}


def available_providers() -> set[str]:
    return set(_PROVIDERS)


def build_runner(provider: str, model: str, **kwargs: Any):
    provider = provider.lower()
    if provider == "openai":
        from gaslight_detector.runners.openai_runner import OpenAIRunner
        return OpenAIRunner(model, **kwargs)
    if provider == "anthropic":
        from gaslight_detector.runners.anthropic_runner import AnthropicRunner
        return AnthropicRunner(model, **kwargs)
    if provider == "local":
        from gaslight_detector.runners.local_runner import LocalRunner
        return LocalRunner(model, **kwargs)
    if provider == "replay":
        from gaslight_detector.runners.replay_runner import ReplayRunner
        return ReplayRunner(model, **kwargs)
    raise ValueError(f"Unknown provider {provider!r}. Known: {sorted(_PROVIDERS)}.")
