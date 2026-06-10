from __future__ import annotations

from typing import Any

from gaslight_detector.config import ScoringConfig
from gaslight_detector.scoring.embedders.base import TextEmbedder, get_embedder
from gaslight_detector.scoring.scorers.base import Scorer
from gaslight_detector.scoring.scorers.judge_scorer import JudgeScorer
from gaslight_detector.scoring.scorers.lexical_scorer import LexicalScorer
from gaslight_detector.scoring.scorers.semantic_scorer import SemanticScorer


class ScorerPanel:
    """Builds and holds the configured scorers and records their versions.

    `lexical` is always present (it supplies the refusal gate, hedging, and surface shell).
    `semantic` is the default substance signal. `judge` is optional and may be local or
    cross-provider. The panel exposes `versions()` so reports can list every scorer + version
    separately — substance numbers are never merged without their provenance.
    """

    def __init__(self, cfg: ScoringConfig, embedder: TextEmbedder | None = None,
                 judge_runner: Any | None = None) -> None:
        self.cfg = cfg
        self.scorers: list[Scorer] = []
        self._embedder = embedder

        if "semantic" in cfg.scorers and self._embedder is None:
            self._embedder = get_embedder(cfg.embedder, embedding_model=cfg.embedding_model)

        # lexical is mandatory and first
        self.scorers.append(LexicalScorer(cfg))
        if "semantic" in cfg.scorers:
            self.scorers.append(SemanticScorer(self._embedder))
        if "judge" in cfg.scorers:
            if judge_runner is None:
                raise ValueError("judge scorer configured but no judge runner was provided.")
            self.scorers.append(JudgeScorer(judge_runner))

    @property
    def embedder(self) -> TextEmbedder | None:
        return self._embedder

    def versions(self) -> list[dict[str, Any]]:
        return [{"name": s.name, "version": s.version, "learned": bool(getattr(s, "learned", False))}
                for s in self.scorers]

    def score_all(self, response: str, prompt: str, task: Any) -> dict[str, Any]:
        return {s.name: s.score(response, prompt, task) for s in self.scorers}
