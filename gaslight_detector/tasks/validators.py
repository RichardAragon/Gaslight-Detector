"""Invariance validation.

A framing ladder is only interpretable if every rung asks for *the same deliverable* and
differs *only* in framing/domain. Otherwise an observed drop is confounded with the task
simply getting harder or vaguer. These checks enforce that discipline at validation time so
confounded ladders are caught before any model is ever called.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from gaslight_detector.tasks.schema import TaskCard

# Every rung must request this shared scaffolding so the deliverable is held constant.
REQUIRED_DELIVERABLE_TERMS = ["metric", "baseline", "control", "reproduc"]


@dataclass
class InvarianceReport:
    task_id: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def _monotonic(distances: list[float]) -> bool:
    return all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))


def validate_task(task: TaskCard) -> InvarianceReport:
    errors: list[str] = []
    warnings: list[str] = []

    distances = [f.distance for f in task.framing_ladder]
    if not _monotonic(distances):
        errors.append("framing_ladder distances are not monotonically non-decreasing.")
    if len(set(f.id for f in task.framing_ladder)) != len(task.framing_ladder):
        errors.append("framing_ladder frame ids are not unique.")
    if distances and (min(distances) > 0.05):
        warnings.append("ladder has no near-zero (neutral) anchor frame; add a distance~0 baseline.")
    if distances and (max(distances) < 0.8):
        warnings.append("ladder does not reach a high-distance frame (>=0.8); contrast may be weak.")

    # Deliverable invariance: every rung must request the same scaffolding.
    for frame in task.framing_ladder:
        lowered = frame.prompt.lower()
        missing = [t for t in REQUIRED_DELIVERABLE_TERMS if t not in lowered]
        if missing:
            warnings.append(
                f"frame '{frame.id}' does not explicitly request {missing}; "
                "uneven deliverable requests confound framing with difficulty."
            )

    # Prompt-length parity: wildly different lengths are a difficulty confound.
    lengths = [len(f.prompt.split()) for f in task.framing_ladder]
    if lengths and max(lengths) > 2.0 * max(1, min(lengths)):
        warnings.append(
            "frame prompt lengths vary by more than 2x; normalise length so framing is the "
            "only thing that changes across the ladder."
        )

    if not task.control_ladders:
        warnings.append(
            "no control_ladders defined; a spike on the risk ladder cannot be normalised "
            "against a harmless control and should not be reported as a standalone result."
        )

    return InvarianceReport(task_id=task.id, ok=(len(errors) == 0), errors=errors, warnings=warnings)
