from __future__ import annotations

from typing import Sequence

from gaslight_detector.geometry.distances import (
    cosine,
    semantic_shell_distance,
    structural_distance,
)


def transition_scores(
    feature_dicts: list[dict[str, float]],
    embeddings: Sequence[Sequence[float] | None] | None = None,
    epsilon: float = 1e-3,
) -> list[dict[str, float]]:
    """Per-transition geometry.

    ghost_ratio = structural change / semantic-shell change. A high ratio means the technical
    structure moved far while the surface barely moved — the 'silent' signature.

    semantic_continuity (when embeddings are present) is the cosine similarity of consecutive
    answers. It separates 'the model changed topic' (low continuity) from 'the model stayed on
    topic but hollowed out' (high continuity + high structural change) — the case we care about.
    """
    out: list[dict[str, float]] = []
    for i in range(len(feature_dicts) - 1):
        structural = structural_distance(feature_dicts[i], feature_dicts[i + 1])
        semantic = semantic_shell_distance(feature_dicts[i], feature_dicts[i + 1])
        row = {
            "transition_index": i,
            "structural_distance": structural,
            "semantic_shell_distance": semantic,
            "ghost_ratio": structural / (semantic + epsilon),
        }
        if embeddings is not None:
            cont = cosine(embeddings[i], embeddings[i + 1])
            row["semantic_continuity"] = cont if cont is not None else float("nan")
        out.append(row)
    return out
