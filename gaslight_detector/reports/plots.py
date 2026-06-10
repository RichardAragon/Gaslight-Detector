from __future__ import annotations

from pathlib import Path
from typing import Any

from gaslight_detector.logging_utils import get_logger

log = get_logger(__name__)


def write_plots(result: dict[str, Any], out_dir: str | Path) -> list[str]:
    """Render plots if matplotlib is available; otherwise skip gracefully."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        log.info("matplotlib not installed; skipping plots.")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    risk = result["risk_ladder"]
    d = risk["distances"]
    written: list[str] = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d, risk["mps_by_frame"], marker="o", label="MPS (utility)")
    ax.plot(d, risk["composite_substance_by_frame"], marker="s", label="Composite substance")
    ax.plot(d, risk["hedging_by_frame"], marker="^", linestyle="--", label="Hedging (diagnostic)")
    ax.set_xlabel("Framing distance"); ax.set_ylabel("Score"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Framing curve"); ax.legend(); fig.tight_layout()
    p1 = "framing_curve.png"; fig.savefig(out_dir / p1, dpi=130); plt.close(fig); written.append(p1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d, risk["semantic_shell_by_frame"], marker="o", label="Semantic shell (fluency)")
    ax.plot(d, risk["lexical_substance_by_frame"], marker="s", label="Lexical substance (baseline)")
    ax.fill_between(d, risk["composite_substance_by_frame"], risk["semantic_shell_by_frame"],
                    alpha=0.15, color="red")
    ax.set_xlabel("Framing distance"); ax.set_ylabel("Score"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Semantic shell vs structural mass"); ax.legend(); fig.tight_layout()
    p2 = "semantic_shell_vs_structural_mass.png"
    fig.savefig(out_dir / p2, dpi=130); plt.close(fig); written.append(p2)
    return written
