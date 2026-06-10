from __future__ import annotations

from typing import Any

from gaslight_detector.runners.base import BaseRunner


class AnthropicRunner(BaseRunner):
    provider_name = "anthropic"

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        try:
            import anthropic
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install providers: `pip install gaslight-detector[providers]`.") from exc
        self._client = anthropic.Anthropic()

    def _generate_once(self, prompt: str, system: str | None = None) -> str:  # pragma: no cover
        kwargs: dict[str, Any] = dict(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        msg = self._client.messages.create(**kwargs)
        return "\n".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
