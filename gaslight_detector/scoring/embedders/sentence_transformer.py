from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    """Learned sentence embeddings. Optional dependency: sentence-transformers."""
    learned = True

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import sentence_transformers as st
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Semantic embedding requires `pip install gaslight-detector[embedding]`."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.version = getattr(st, "__version__", "unknown")

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - heavy dep
        vecs = self._model.encode([t or " " for t in texts], normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)
