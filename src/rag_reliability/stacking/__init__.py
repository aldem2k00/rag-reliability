"""Стэкинг разнородных сигналов над готовыми артефактами."""

from rag_reliability.stacking.collect import collect_features
from rag_reliability.stacking.stack import (
    FEATURE_SET_V1,
    fit_stack,
    make_prediction_fit_fn,
    select_features_by_ci,
)

__all__ = [
    "FEATURE_SET_V1",
    "collect_features",
    "fit_stack",
    "make_prediction_fit_fn",
    "select_features_by_ci",
]
