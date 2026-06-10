from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from gaslight_detector.runners.base import BaseRunner


class ReplayRunner(BaseRunner):
    """Replays previously saved raw responses keyed by (ladder, frame_id). Enables fully
    deterministic, offline re-scoring with zero provider calls — the backbone of reproducibility.
    """
    provider_name = "replay"

    def __init__(self, model: str, replay_file: str | Path, **kwargs: Any) -> None:
        kwargs.pop("cache", None)  # replay is inherently deterministic; never cache
        super().__init__(model=model, cache=False, **kwargs)
        self.replay_file = Path(replay_file)
        self._by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._cursor: dict[tuple[str, str], int] = defaultdict(int)
        for line in self.replay_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec.get("ladder", "risk"), rec["frame_id"])
            self._by_key[key].append(rec.get("response", ""))
        self._active_key: tuple[str, str] | None = None

    def sample_count(self, ladder: str, frame_id: str) -> int:
        return len(self._by_key.get((ladder, frame_id), []))

    def counts(self) -> dict:
        return {f"{k[0]}/{k[1]}": len(v) for k, v in self._by_key.items()}

    def set_frame(self, ladder: str, frame_id: str) -> None:
        self._active_key = (ladder, frame_id)

    def _generate_once(self, prompt: str, system: str | None = None) -> str:
        if self._active_key is None:
            raise RuntimeError("ReplayRunner.set_frame must be called before generate.")
        bucket = self._by_key.get(self._active_key, [])
        i = self._cursor[self._active_key]
        if i >= len(bucket):
            # cycle if fewer saved samples than requested
            if not bucket:
                raise RuntimeError(f"No replay responses for {self._active_key}.")
            return bucket[i % len(bucket)]
        self._cursor[self._active_key] += 1
        return bucket[i]
