from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from gaslight_detector.analysis.bootstrap import bootstrap_max_drop
from gaslight_detector.config import Config
from gaslight_detector.geometry.discontinuity import transition_scores
from gaslight_detector.geometry.spike import detect_spike
from gaslight_detector.scoring.aggregate import FrameScore, score_response
from gaslight_detector.scoring.panel import ScorerPanel
from gaslight_detector.tasks.schema import Frame, TaskCard
from gaslight_detector.version import RESULT_SCHEMA_VERSION, __version__


@dataclass
class LadderAnalysis:
    name: str
    frame_ids: list[str]
    distances: list[float]
    mps_by_frame: list[float]
    mps_samples: list[list[float]]
    composite_substance_by_frame: list[float]
    lexical_substance_by_frame: list[float]
    scorer_substance_by_frame: dict[str, list[float]]
    semantic_shell_by_frame: list[float]
    hedging_by_frame: list[float]
    fulfillment_by_frame: list[float | None]
    transitions: list[dict[str, float]]
    spike: dict[str, Any]
    bootstrap: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class AnalysisResult:
    schema_version: str
    gaslight_detector_version: str
    timestamp: str
    provider: str
    model: str
    task: dict[str, Any]
    config: dict[str, Any]
    scorers: list[dict[str, Any]]
    risk_ladder: dict[str, Any]
    controls: dict[str, Any] = field(default_factory=dict)
    differential: dict[str, Any] = field(default_factory=dict)
    exploratory_only: bool = True
    exploratory_reason: str = ""
    disclaimer: str = (
        "Gaslight Detector measures framing-conditioned structural deformation. It does not infer "
        "provider motive or prove intentional degradation. A flagged spike is a hypothesis to "
        "investigate with the prescribed controls, not evidence of intent."
    )
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def _mean_frame_score(scores: list[FrameScore]) -> FrameScore:
    if len(scores) == 1:
        return scores[0]
    sub_names = scores[0].scorer_substance.keys()
    scorer_substance = {n: float(np.mean([s.scorer_substance[n] for s in scores])) for n in sub_names}
    comp_names = scores[0].scorer_components.keys()
    scorer_components = {
        n: {k: float(np.mean([s.scorer_components[n].get(k, 0.0) for s in scores]))
            for k in scores[0].scorer_components[n].keys()}
        for n in comp_names
    }
    fulfil = [s.task_fulfillment for s in scores if s.task_fulfillment is not None]
    emb_list = [s.embedding for s in scores if s.embedding is not None]
    embedding = list(np.mean(emb_list, axis=0)) if emb_list else None
    return FrameScore(
        scorer_substance=scorer_substance,
        scorer_components=scorer_components,
        structural_mass_lexical=float(np.mean([s.structural_mass_lexical for s in scores])),
        composite_substance=float(np.mean([s.composite_substance for s in scores])),
        semantic_shell=float(np.mean([s.semantic_shell for s in scores])),
        hedging_index=float(np.mean([s.hedging_index for s in scores])),
        task_fulfillment=(float(np.mean(fulfil)) if fulfil else None),
        refusal=float(np.mean([s.refusal for s in scores])),
        manifold_preservation=float(np.mean([s.manifold_preservation for s in scores])),
        primary_substance_learned=scores[0].primary_substance_learned,
        embedding=embedding,
    )


def _score_ladder(frames: list[Frame], responses_by_frame: list[list[str]],
                  task: TaskCard, cfg: Config, panel: ScorerPanel) -> LadderAnalysis:
    mps_samples: list[list[float]] = []
    rep_scores: list[FrameScore] = []
    embeddings: list[list[float] | None] = []

    for frame, responses in zip(frames, responses_by_frame):
        sample_scores = [score_response(r, frame.prompt, task, cfg.scoring, panel) for r in responses]
        mps_samples.append([s.manifold_preservation for s in sample_scores])
        rep = _mean_frame_score(sample_scores)
        rep_scores.append(rep)
        embeddings.append(rep.embedding)

    feature_dicts = [fs.scorer_components.get("lexical", {}) for fs in rep_scores]
    mps_by_frame = [float(np.mean(s)) if s else 0.0 for s in mps_samples]

    boot = bootstrap_max_drop(mps_samples, iterations=cfg.spike.bootstrap_iterations,
                              confidence=cfg.spike.confidence, seed=cfg.runner.seed)
    noise_floor = max(cfg.spike.noise_floor, boot["pooled_sample_std"])
    spike = detect_spike(mps_by_frame, min_drop=cfg.spike.min_drop, min_ratio=cfg.spike.min_ratio,
                         noise_floor=noise_floor, max_drop_ci=tuple(boot["ci"]),
                         ci_lower_must_exceed=cfg.spike.ci_lower_must_exceed).to_dict()

    sub_names = rep_scores[0].scorer_substance.keys() if rep_scores else []
    scorer_substance_by_frame = {
        n: [fs.scorer_substance.get(n, 0.0) for fs in rep_scores] for n in sub_names
    }

    return LadderAnalysis(
        name="",
        frame_ids=[f.id for f in frames],
        distances=[f.distance for f in frames],
        mps_by_frame=mps_by_frame,
        mps_samples=mps_samples,
        composite_substance_by_frame=[fs.composite_substance for fs in rep_scores],
        lexical_substance_by_frame=[fs.structural_mass_lexical for fs in rep_scores],
        scorer_substance_by_frame=scorer_substance_by_frame,
        semantic_shell_by_frame=[fs.semantic_shell for fs in rep_scores],
        hedging_by_frame=[fs.hedging_index for fs in rep_scores],
        fulfillment_by_frame=[fs.task_fulfillment for fs in rep_scores],
        transitions=transition_scores(feature_dicts, embeddings=embeddings),
        spike=spike,
        bootstrap=boot,
    )


def analyze(task: TaskCard, risk_responses: list[list[str]],
            control_responses: dict[str, list[list[str]]], cfg: Config, panel: ScorerPanel,
            provider: str, model: str) -> AnalysisResult:
    risk = _score_ladder(task.framing_ladder, risk_responses, task, cfg, panel)
    risk.name = "risk"

    controls: dict[str, Any] = {}
    for kind, frames in task.control_ladders.items():
        resp = control_responses.get(kind)
        if not resp:
            continue
        c = _score_ladder(frames, resp, task, cfg, panel)
        c.name = kind
        controls[kind] = c.to_dict()

    differential: dict[str, Any] = {}
    exploratory_only = True
    reason = ("No harmless control ladder was scored. Per the methodology, a risk-ladder spike "
              "without a harmless control is exploratory only and supports no topic-selective "
              "conclusion.")
    harmless = controls.get("harmless")
    if harmless is not None:
        risk_md = risk.spike["max_drop"]
        ctrl_md = harmless["spike"]["max_drop"]
        topic_selective = (risk_md - ctrl_md) >= cfg.spike.min_drop
        differential = {
            "risk_max_drop": risk_md,
            "harmless_max_drop": ctrl_md,
            "differential_max_drop": risk_md - ctrl_md,
            "topic_selective": bool(topic_selective),
            "interpretation": (
                "topic-selective deformation (risk ladder drops more than harmless control)"
                if topic_selective
                else "no topic-selective effect beyond general prompt sensitivity"
            ),
        }
        exploratory_only = False
        reason = ""

    return AnalysisResult(
        schema_version=RESULT_SCHEMA_VERSION,
        gaslight_detector_version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
        provider=provider,
        model=model,
        task={"id": task.id, "title": task.title, "domain": task.domain,
              "risk_axis": task.risk_axis, "task_objective": task.task_objective},
        config=cfg.to_dict(),
        scorers=panel.versions(),
        risk_ladder=risk.to_dict(),
        controls=controls,
        differential=differential,
        exploratory_only=exploratory_only,
        exploratory_reason=reason,
        synthetic=task.synthetic,
    )
