from __future__ import annotations

import re

import numpy as np

from gaslight_detector.version import __version__

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Dependency-free, deterministic embedder over hashed word + character n-grams.

    This exists so the semantic pipeline (coverage, continuity) runs offline and tests
    deterministically. It is explicitly NOT a learned model: it captures lexical/morphological
    overlap with some sub-word generalisation, but it does not capture true synonymy. For serious
    or published results use the sentence-transformer embedder. `learned = False` is surfaced in
    reports so this is never mistaken for semantic ground truth.
    """
    name = "hashing-ngram"
    version = __version__
    learned = False

    def __init__(self, dim: int = 1024, char_ngrams: tuple[int, int] = (3, 5)) -> None:
        self.dim = dim
        self.char_lo, self.char_hi = char_ngrams

    def _features(self, text: str) -> list[str]:
        text = text.lower()
        words = _TOKEN.findall(text)
        feats: list[str] = [f"w:{w}" for w in words]
        # word bigrams
        feats += [f"b:{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        # character n-grams over the concatenated token stream (sub-word generalisation)
        joined = " " + " ".join(words) + " "
        for n in range(self.char_lo, self.char_hi + 1):
            feats += [f"c{n}:{joined[i:i+n]}" for i in range(len(joined) - n + 1)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f in self._features(t or " "):
                h = hash(f) % self.dim
                sign = 1.0 if (hash("s" + f) & 1) else -1.0
                out[i, h] += sign
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out
