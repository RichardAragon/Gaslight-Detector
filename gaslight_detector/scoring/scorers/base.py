from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ScorerResult:
    name: str
    version: str
    learned: bool                 # is this backed by a learned model?
    substance: float | None       # 0..1 substance estimate (None if scorer measures only fulfilment)
    fulfillment: float | None = None
    refusal: float | None = None  # only the lexical scorer detects refusal
    hedging: float | None = None
    semantic_shell: float | None = None
    embedding: list[float] | None = None
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("embedding", None)
        return d


class Scorer(Protocol):
    name: str
    version: str
    learned: bool

    def score(self, response: str, prompt: str, task: Any) -> ScorerResult: ...
