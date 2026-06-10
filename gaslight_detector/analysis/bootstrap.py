from __future__ import annotations

import numpy as np


def bootstrap_max_drop(
    frame_samples: list[list[float]],
    iterations: int = 2000,
    confidence: float = 0.90,
    seed: int = 0,
) -> dict:
    """Bootstrap the distribution of the largest single-transition drop.

    frame_samples[i] is the list of per-sample MPS values for frame i. We resample within each
    frame, recompute frame means, recompute consecutive drops, take the max, and repeat. The
    returned CI tells us whether the observed spike survives run-to-run sampling noise.

    Also returns a pooled per-sample standard deviation, used downstream as a principled noise
    floor for the robust dispersion baseline (replacing the fixed 0.03 magic number).
    """
    rng = np.random.default_rng(seed)
    n_frames = len(frame_samples)
    means = np.array([float(np.mean(s)) if s else 0.0 for s in frame_samples])
    point_drops = means[:-1] - means[1:]
    point_max_drop = float(point_drops.max()) if len(point_drops) else 0.0

    # pooled within-frame std (sampling noise)
    stds = [float(np.std(s, ddof=1)) for s in frame_samples if len(s) > 1]
    pooled_std = float(np.mean(stds)) if stds else 0.0

    have_replicates = any(len(s) > 1 for s in frame_samples)
    if not have_replicates:
        # Degenerate: a single sample per frame -> CI is the point estimate.
        return {
            "point_max_drop": point_max_drop,
            "ci": [point_max_drop, point_max_drop],
            "pooled_sample_std": pooled_std,
            "iterations": 0,
        }

    maxdrops = np.empty(iterations)
    for it in range(iterations):
        resampled = np.array([
            float(np.mean(rng.choice(s, size=len(s), replace=True))) if s else 0.0
            for s in frame_samples
        ])
        d = resampled[:-1] - resampled[1:]
        maxdrops[it] = d.max() if len(d) else 0.0

    lo = float(np.percentile(maxdrops, 100 * (1 - confidence) / 2))
    hi = float(np.percentile(maxdrops, 100 * (1 + confidence) / 2))
    return {
        "point_max_drop": point_max_drop,
        "ci": [lo, hi],
        "pooled_sample_std": pooled_std,
        "iterations": iterations,
    }
