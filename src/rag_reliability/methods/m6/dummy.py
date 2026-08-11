"""Deterministic stub NLI/embedder so the m6 pipeline runs without torch."""

from __future__ import annotations

import hashlib

import numpy as np


def _unit_float(*parts: str) -> float:
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class DummyNLI:
    """Mirror NLIScorer.score with deterministic entailment scores from text hashes."""

    def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        output: list[dict[str, float]] = []
        for premise, hypothesis in pairs:
            if premise == hypothesis:
                output.append({"entail": 1.0, "contra": 0.0})
                continue
            entail = _unit_float("entail", premise, hypothesis)
            contra = (1.0 - entail) * _unit_float("contra", premise, hypothesis)
            output.append({"entail": entail, "contra": contra})
        return output


class DummyEmbedder:
    """Mirror SentenceTransformer.encode with deterministic normalized vectors."""

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,  # noqa: ARG002
    ) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "big")
            vector = np.random.default_rng(seed).normal(size=32)
            vectors.append(vector / np.linalg.norm(vector))
        return np.stack(vectors)
