from __future__ import annotations

import hashlib
import time
from typing import Any

from gaslight_detector.logging_utils import get_logger

log = get_logger(__name__)


class BaseRunner:
    """Provider-agnostic generation with retry/backoff, optional in-memory caching, and
    multi-sample support. Subclasses implement `_generate_once`.
    """
    provider_name = "base"

    def __init__(self, model: str, *, temperature: float = 0.2, max_tokens: int = 2000,
                 max_retries: int = 4, retry_base_delay: float = 1.0, cache: bool = True,
                 **kwargs: Any) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._cache_enabled = cache
        self._cache: dict[str, str] = {}
        self.extra = kwargs

    def _generate_once(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError

    def generate(self, prompt: str, system: str | None = None, cache_salt: Any = None) -> str:
        """Generate one completion.

        `cache_salt` is mixed into the cache key ONLY. It never alters the prompt sent to the
        model. This lets multiple stochastic samples of the *same* prompt be cached as distinct
        entries without contaminating the input the way an inline marker would.
        """
        key = None
        if self._cache_enabled:
            key = hashlib.sha256(
                f"{self.provider_name}|{self.model}|{system}|{cache_salt}|{prompt}".encode("utf-8")
            ).hexdigest()
            if key in self._cache:
                return self._cache[key]

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                out = self._generate_once(prompt, system=system)
                if self._cache_enabled and key is not None:
                    self._cache[key] = out
                return out
            except Exception as exc:  # pragma: no cover - network paths
                last_exc = exc
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                log.warning("generate attempt %d/%d failed: %s (retry in %.1fs)",
                            attempt, self.max_retries, exc, delay)
                time.sleep(delay)
        raise RuntimeError(f"Generation failed after {self.max_retries} attempts.") from last_exc

    def sample(self, prompt: str, n: int, system: str | None = None) -> list[str]:
        """Draw n samples of the SAME prompt.

        The prompt text is never modified between samples (the v0.2 inline `<<sample i>>` marker
        is gone — it leaked into the model input). Distinct samples are kept apart only via the
        cache salt. For a deterministic provider (temperature == 0, non-replay) the output cannot
        vary, so we make a single real call and replicate it rather than burning n identical calls.
        """
        n = max(1, n)
        if self.temperature == 0 and self.provider_name != "replay":
            return [self.generate(prompt, system=system, cache_salt=0)] * n
        return [self.generate(prompt, system=system, cache_salt=i) for i in range(n)]
