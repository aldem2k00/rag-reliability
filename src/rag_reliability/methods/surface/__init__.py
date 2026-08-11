"""Не-LLM бейзлайн на поверхностных фичах: перенос из ветки m3-m6."""

from rag_reliability.methods.surface.features import (
    FEATURE_KEYS,
    ngram_overlap,
    split_chunks,
    surface_features,
)

__all__ = ["FEATURE_KEYS", "ngram_overlap", "split_chunks", "surface_features"]
