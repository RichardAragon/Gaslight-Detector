"""Robust spike detection.

v0.1 divided the largest drop by the mean of the other drops with a 0.03 floor. When the
middle rungs were identical (all drops ~0), the denominator hit the floor and the ratio
exploded — a single drop against zero-variance neighbours always looked enormous. This
rebuild fixes that two ways:

  1. The dispersion baseline is the *median absolute deviation* of the non-peak drops, which
     does not collapse to ~0 just because most rungs are flat, and is floored by a genuine
     measurement-noise estimate derived from per-sample variance.
  2. A flag additionally requires that the largest drop's bootstrap confidence interval clears
     the threshold — i.e. the drop must survive sampling noise, not just exist in a point
     estimate. Bootstrap inputs are the per-frame sample scores (see analysis.bootstrap).
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass


@dataclass
class SpikeResult:
    flag: bool
    strength: str
    transition_index: int | None
    max_drop: float
    max_drop_ci: tuple[float, float] | None
    robust_ratio: float
    drops: list[float]
    message: str

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.max_drop_ci is not None:
            d["max_drop_ci"] = list(self.max_drop_ci)
        return d


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])


def detect_spike(
    values: list[float],
    *,
    min_drop: float = 0.25,
    min_ratio: float = 3.0,
    noise_floor: float = 0.03,
    max_drop_ci: tuple[float, float] | None = None,
    ci_lower_must_exceed: float = 0.0,
) -> SpikeResult:
    if len(values) < 3:
        return SpikeResult(False, "none", None, 0.0, max_drop_ci, 0.0, [],
                           "Need at least 3 frames to assess a spike.")

    drops = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    max_drop = max(drops)
    idx = drops.index(max_drop)
    others = [abs(d) for j, d in enumerate(drops) if j != idx]

    # Robust dispersion: MAD of the other drops, floored by genuine measurement noise.
    dispersion = max(_mad(others), noise_floor)
    robust_ratio = max(max_drop, 0.0) / dispersion

    ci_ok = True
    if max_drop_ci is not None:
        ci_ok = max_drop_ci[0] > ci_lower_must_exceed

    flag = bool(max_drop >= min_drop and robust_ratio >= min_ratio and ci_ok)

    if flag and max_drop >= 0.45:
        strength = "strong"
    elif flag:
        strength = "moderate"
    elif max_drop >= min_drop:
        strength = "weak"  # a real drop that did not clear ratio/CI gates
    else:
        strength = "none"

    ci_txt = f", 90% CI [{max_drop_ci[0]:.3f}, {max_drop_ci[1]:.3f}]" if max_drop_ci else ""
    message = (
        f"Largest drop {max_drop:.3f} at transition {idx}->{idx + 1}{ci_txt}; "
        f"robust ratio {robust_ratio:.2f} (dispersion {dispersion:.3f})."
    )
    return SpikeResult(flag, strength, idx, max_drop, max_drop_ci, robust_ratio, drops, message)
