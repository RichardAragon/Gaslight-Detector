from __future__ import annotations

from typing import Any

from gaslight_detector.runners.base import BaseRunner


class LocalRunner(BaseRunner):
    """Local HuggingFace Transformers runner. Useful as an open-weights control alongside
    hosted models (recommended control #4 in the README)."""
    provider_name = "local"

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install local model deps: `pip install gaslight-detector[local]`.") from exc
        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")

    def _generate_once(self, prompt: str, system: str | None = None) -> str:  # pragma: no cover
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=self.max_tokens, temperature=self.temperature,
                                   do_sample=self.temperature > 0)
        return self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
