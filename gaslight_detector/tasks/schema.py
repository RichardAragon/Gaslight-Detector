from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Frame:
    id: str
    distance: float
    prompt: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.distance <= 1.0:
            raise ValueError(f"Frame {self.id}: distance must be in [0, 1], got {self.distance}.")
        if not self.prompt.strip():
            raise ValueError(f"Frame {self.id}: prompt is empty.")


@dataclass
class TaskCard:
    id: str
    title: str
    domain: str
    risk_axis: str
    task_objective: str
    invariant_core: list[str]
    concepts: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    causal_edges: list[dict[str, str]] = field(default_factory=list)
    procedure_min_steps: int = 6
    framing_ladder: list[Frame] = field(default_factory=list)
    control_ladders: dict[str, list[Frame]] = field(default_factory=dict)
    synthetic: bool = False

    def expected(self) -> dict[str, Any]:
        return {
            "concepts": self.concepts,
            "metrics": self.metrics,
            "baselines": self.baselines,
            "invariant_core": self.invariant_core,
            "causal_edges": self.causal_edges,
            "procedure_min_steps": self.procedure_min_steps,
        }


def _frames(raw: list[dict[str, Any]]) -> list[Frame]:
    return [Frame(id=f["id"], distance=float(f["distance"]), prompt=str(f["prompt"])) for f in raw]


def load_task(path: str | Path) -> TaskCard:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    esf = data.get("expected_structural_features", {}) or {}
    procedures = esf.get("procedures", {}) or {}
    controls_raw = data.get("control_ladders", {}) or {}

    task = TaskCard(
        id=data["id"],
        title=data["title"],
        domain=data.get("domain", "unspecified"),
        risk_axis=data.get("risk_axis", "unspecified"),
        task_objective=str(data.get("task_objective", "")).strip(),
        invariant_core=list(data.get("invariant_core", [])),
        concepts=list(esf.get("concepts", [])),
        metrics=list(esf.get("metrics", [])),
        baselines=list(esf.get("baselines", [])),
        causal_edges=list(esf.get("causal_edges", [])),
        procedure_min_steps=int(procedures.get("min_steps", 6) or 6),
        framing_ladder=_frames(data.get("framing_ladder", [])),
        control_ladders={k: _frames(v) for k, v in controls_raw.items()},
        synthetic=bool(data.get("synthetic", False)),
    )
    if len(task.framing_ladder) < 3:
        raise ValueError(f"Task {task.id}: framing_ladder needs >= 3 frames.")
    if not task.invariant_core:
        raise ValueError(f"Task {task.id}: invariant_core must not be empty.")
    return task
