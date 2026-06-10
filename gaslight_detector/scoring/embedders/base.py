from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from gaslight_detector.logging_utils import get_logger

log = get_logger(__name__)


class TextEmbedder(Protocol):
    name: str
    version: str
    learned: bool  # True for a learned model; False for the lexical fallback

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) array of L2-normalised, non-negative-or-signed embeddings."""
        ...


def get_embedder(choice: str = "auto", embedding_model: str | None = None, **kwargs: Any) -> "TextEmbedder":
    """Resolve an embedder.

    'auto'                -> sentence-transformers if importable, else the hashing fallback
    'sentence-transformer'-> sentence-transformers (hard error if missing)
    'hashing'             -> deterministic, dependency-free fallback (NOT a learned model)
    """
    choice = (choice or "auto").lower()
    if choice in ("sentence-transformer", "sentence_transformers", "st"):
        from gaslight_detector.scoring.embedders.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(embedding_model or "sentence-transformers/all-MiniLM-L6-v2")
    if choice == "hashing":
        from gaslight_detector.scoring.embedders.hashing import HashingEmbedder
        return HashingEmbedder()
    # auto
    try:
        from gaslight_detector.scoring.embedders.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(embedding_model or "sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        from gaslight_detector.scoring.embedders.hashing import HashingEmbedder
        log.warning(
            "sentence-transformers not available; falling back to the HASHING embedder. "
            "This is a deterministic n-gram approximation, NOT a learned semantic model. "
            "Install `gaslight-detector[embedding]` for true paraphrase robustness."
        )
        return HashingEmbedder()
