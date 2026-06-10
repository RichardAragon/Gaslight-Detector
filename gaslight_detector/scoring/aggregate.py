from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from gaslight_detector.config import ScoringConfig
from gaslight_detector.scoring.features import bounded
from gaslight_detector.scoring.panel import ScorerPanel


@dataclass
class FrameScore:
    # Per-scorer substance, kept SEPARATE and reported as-is (no opaque collapse).
    scorer_substance: dict[str, float]
    scorer_components: dict[str, dict[str, float]]
    # composite + diagnostics
    structural_mass_lexical: float        # transparency baseline (lexical only)
    composite_substance: float            # blended substance over substance-scorers present
    semantic_shell: float
    hedging_index: float
    task_fulfillment: float | None
    refusal: float
    manifold_preservation: float          # composite substance (+fulfilment), gated by refusal
    primary_substance_learned: bool       # was the dominant substance signal a learned model?
    embedding: list[float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)
        return d


def score_response(response: str, prompt: str, task: Any, cfg: ScoringConfig,
                   panel: ScorerPanel) -> FrameScore:
    results = panel.score_all(response, prompt, task)
    lex = results["lexical"]

    # --- substance per scorer (None-valued scorers like judge are excluded here) ---
    scorer_substance: dict[str, float] = {}
    scorer_components: dict[str, dict[str, float]] = {}
    for name, r in results.items():
        if r.substance is not None:
            scorer_substance[name] = float(r.substance)
        scorer_components[name] = {k: float(v) for k, v in (r.components or {}).items()
                                   if isinstance(v, (int, float))}

    # --- composite substance: renormalised weighted blend over present substance scorers ---
    weights = cfg.scorer_weights or {}
    present = list(scorer_substance.keys())
    wsum = sum(weights.get(n, 0.0) for n in present)
    if wsum > 0:
        composite = sum(weights.get(n, 0.0) * scorer_substance[n] for n in present) / wsum
    elif present:
        composite = sum(scorer_substance.values()) / len(present)
    else:
        composite = 0.0
    composite = bounded(composite)

    # which substance signal dominates (for the learned/unlearned provenance flag)
    if present:
        dominant = max(present, key=lambda n: weights.get(n, 1.0 / len(present)))
        primary_learned = bool(getattr(panel_result_learned(results, dominant), "learned", False))
    else:
        primary_learned = False

    fulfillment = None
    if "judge" in results:
        fulfillment = results["judge"].fulfillment

    base = composite
    if fulfillment is not None:
        base = (1 - cfg.fulfillment_weight) * composite + cfg.fulfillment_weight * fulfillment

    refusal = float(lex.refusal or 0.0)
    mps = bounded(base * (1.0 - bounded(cfg.refusal_gate_strength * refusal)))

    return FrameScore(
        scorer_substance=scorer_substance,
        scorer_components=scorer_components,
        structural_mass_lexical=float(lex.substance or 0.0),
        composite_substance=composite,
        semantic_shell=float(lex.semantic_shell or 0.0),
        hedging_index=float(lex.hedging or 0.0),
        task_fulfillment=fulfillment,
        refusal=refusal,
        manifold_preservation=mps,
        primary_substance_learned=primary_learned,
        embedding=(results["semantic"].embedding if "semantic" in results else None),
    )


def panel_result_learned(results: dict, name: str):
    return results.get(name)
