from __future__ import annotations

import math
from typing import Mapping, Sequence


def l2(a: Mapping[str, float], b: Mapping[str, float], keys: list[str]) -> float:
    return math.sqrt(sum((float(a.get(k, 0.0)) - float(b.get(k, 0.0))) ** 2 for k in keys))


_STRUCTURAL_KEYS = [
    "concept", "invariant", "metric", "baseline",
    "procedure", "code", "causal", "causal_edge",
]
_SHELL_KEYS = ["concept", "invariant", "length"]


def structural_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return l2(a, b, _STRUCTURAL_KEYS) / math.sqrt(len(_STRUCTURAL_KEYS))


def semantic_shell_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return l2(a, b, _SHELL_KEYS) / math.sqrt(len(_SHELL_KEYS))


def cosine(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)
