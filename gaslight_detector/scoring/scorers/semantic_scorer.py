from __future__ import annotations

import re
from typing import Any

import numpy as np

from gaslight_detector.scoring.embedders.base import TextEmbedder
from gaslight_detector.scoring.scorers.base import ScorerResult

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT.split(text or "") if s.strip()]
    return parts or [text or " "]


class SemanticScorer:
    """Substance via embedding coverage — the v0.3 default substance signal.

    Instead of asking "does the literal token 'baseline' appear?", it embeds each expected item
    (concept / metric / baseline / invariant requirement) and each response sentence, then scores
    each item by its best semantic match in the response. With a learned embedder this is robust
    to paraphrase ("control condition" satisfies "baseline"); with the hashing fallback it degrades
    to morphological overlap, which is why the scorer reports `learned` so a reader knows which
    regime produced the number.

    It also returns the response embedding so the analyzer can compute semantic continuity between
    frames using whichever embedder is active.
    """
    name = "semantic"
    learned = True

    def __init__(self, embedder: TextEmbedder) -> None:
        self.embedder = embedder
        self.version = f"{getattr(embedder, 'name', 'embedder')}@{getattr(embedder, 'version', '?')}"
        self.learned = bool(getattr(embedder, "learned", False))

    def _coverage(self, item_vecs: np.ndarray, sent_vecs: np.ndarray) -> float:
        if item_vecs.size == 0 or sent_vecs.size == 0:
            return 0.0
        sims = item_vecs @ sent_vecs.T            # (items, sentences), vectors are L2-normalised
        best = sims.max(axis=1)                    # best matching sentence per expected item
        best = np.clip(best, 0.0, 1.0)
        return float(best.mean())

    def score(self, response: str, prompt: str, task: Any) -> ScorerResult:
        expected = task.expected()
        groups = {
            "concepts": expected.get("concepts", []),
            "metrics": expected.get("metrics", []),
            "baselines": expected.get("baselines", []),
            "invariant_core": expected.get("invariant_core", []),
        }
        sents = _sentences(response)
        sent_vecs = self.embedder.embed(sents)
        resp_vec = self.embedder.embed([response or " "])[0]

        weights = {"concepts": 0.30, "metrics": 0.25, "baselines": 0.20, "invariant_core": 0.25}
        comp: dict[str, float] = {}
        total_w = 0.0
        acc = 0.0
        for g, items in groups.items():
            if not items:
                continue
            cov = self._coverage(self.embedder.embed(items), sent_vecs)
            comp[g] = cov
            acc += weights[g] * cov
            total_w += weights[g]
        substance = float(acc / total_w) if total_w else 0.0
        return ScorerResult(
            name=self.name, version=self.version, learned=self.learned,
            substance=substance, components=comp,
            embedding=[float(x) for x in resp_vec],
        )
