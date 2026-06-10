from __future__ import annotations

from typing import Any

from gaslight_detector.config import ScoringConfig
from gaslight_detector.scoring.features import bounded, extract_lexical_features
from gaslight_detector.scoring.scorers.base import ScorerResult
from gaslight_detector.version import __version__


def _structural_mass(f, w) -> float:
    return bounded(
        w.concept * f.concept + w.invariant * f.invariant + w.metric * f.metric
        + w.baseline * f.baseline + w.procedure * f.procedure + w.code * f.code
        + w.causal * f.causal + w.causal_edge * f.causal_edge
    )


class LexicalScorer:
    """Transparent keyword/structure baseline. Always run, always reported separately.

    In v0.3 this is no longer the primary substance signal — it is the auditable, dependency-free
    floor. It also supplies the refusal gate, the hedging diagnostic, and the surface-fluency
    shell, which the semantic and judge scorers do not measure.
    """
    name = "lexical"
    version = __version__
    learned = False

    def __init__(self, cfg: ScoringConfig) -> None:
        self.cfg = cfg

    def score(self, response: str, prompt: str, task: Any) -> ScorerResult:
        f = extract_lexical_features(response, task.expected())
        mass = _structural_mass(f, self.cfg.weights)
        shell = bounded(0.5 * f.concept + 0.3 * f.invariant + 0.2 * f.length)
        hedging = bounded(self.cfg.hedging_genericness_weight * f.genericness
                          + self.cfg.hedging_safety_weight * f.safety_framing)
        return ScorerResult(
            name=self.name, version=self.version, learned=False,
            substance=mass, refusal=min(1.0, float(f.refusal)),
            hedging=hedging, semantic_shell=shell,
            components={k: float(v) for k, v in f.to_dict().items()},
        )
