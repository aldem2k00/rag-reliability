"""Обучение стэка и статистический отбор его признаков."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rag_reliability.schema import Prediction

FEATURE_SET_V1 = [
    "surf.p_faith",
    "surf.p_rel",
    "m3.p_faith",
    "enc.logit",
    "ld.max_unsup",
]

_LOGREG_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class StackModel:
    """Калиброванная модель с явной записью фактически выбранной калибровки."""

    estimator: Any
    calibration_method: str

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        features = _validate_matrix(matrix)
        probabilities = np.asarray(self.estimator.predict_proba(features), dtype=float)
        if probabilities.shape != (len(features), 2):
            raise ValueError(
                "Stack estimator must return two class probabilities per row, "
                f"got {probabilities.shape}"
            )
        return probabilities


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    features = np.asarray(matrix, dtype=float)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError(
            f"X must be a two-dimensional matrix with at least one feature, got {features.shape}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("X contains non-finite feature values")
    return features


def _calibration_method(calibrate: str, n_train: int) -> str:
    if calibrate == "auto":
        return "isotonic" if n_train >= 500 else "sigmoid"
    if calibrate in {"platt", "sigmoid"}:
        return "sigmoid"
    if calibrate == "isotonic":
        return "isotonic"
    raise ValueError(
        f"calibrate must be 'auto', 'platt', 'sigmoid' or 'isotonic', got {calibrate!r}"
    )


def _base_estimator(model: str, seed: int, inner_cv: StratifiedKFold) -> Any:
    if model == "logreg":
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        return GridSearchCV(
            pipeline,
            {"model__C": list(_LOGREG_C_GRID)},
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=1,
            refit=True,
        )
    if model == "hgb":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"model must be 'logreg' or 'hgb', got {model!r}")


def fit_stack(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str = "logreg",
    calibrate: str = "auto",
    seed: int = 0,
) -> StackModel:
    """Обучить scaler, классификатор и калибратор только на переданном train.

    Порог здесь намеренно отсутствует: его отдельно подбирает протокол внутри
    train-части каждого внешнего фолда.
    """
    features = _validate_matrix(X)
    labels = np.asarray(y, dtype=int)
    if labels.shape != (len(features),):
        raise ValueError(
            f"y must contain one label per feature row, got {labels.shape} for {len(features)} rows"
        )
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("y must contain only binary labels 0 and 1")
    counts = np.bincount(labels, minlength=2)
    if int(np.min(counts)) < 4:
        raise ValueError(
            "fit_stack needs at least four cases of each class for nested CV, "
            f"got class counts {counts.tolist()}"
        )

    n_splits = min(3, int(np.min(counts)))
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    calibration_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed + 1,
    )
    calibration_method = _calibration_method(calibrate, len(features))
    estimator = CalibratedClassifierCV(
        estimator=_base_estimator(model, seed, inner_cv),
        method=calibration_method,
        cv=calibration_cv,
    )
    estimator.fit(features, labels)
    return StackModel(
        estimator=estimator,
        calibration_method=calibration_method,
    )


def _prediction_matrix(
    predictions: Sequence[Prediction],
    feature_keys: Sequence[str],
) -> np.ndarray:
    rows: list[list[float]] = []
    for prediction in predictions:
        missing = [key for key in feature_keys if key not in prediction.scores]
        if missing:
            raise ValueError(
                f"Prediction {prediction.id!r} misses {len(missing)} stack feature(s): "
                f"{missing[:5]}"
            )
        rows.append([float(prediction.scores[key]) for key in feature_keys])
    return _validate_matrix(np.asarray(rows, dtype=float))


def make_prediction_fit_fn(
    feature_keys: Sequence[str],
    *,
    model: str = "logreg",
    calibrate: str = "auto",
    seed: int = 0,
) -> Callable[
    [Sequence[Prediction], np.ndarray],
    Callable[[Sequence[Prediction]], np.ndarray],
]:
    """Адаптировать ``fit_stack`` к fit_fn-контракту внешней CV."""
    keys = tuple(feature_keys)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("feature_keys must be non-empty and unique")

    def fit_predictions(
        train_predictions: Sequence[Prediction],
        y_train: np.ndarray,
    ) -> Callable[[Sequence[Prediction]], np.ndarray]:
        fitted = fit_stack(
            _prediction_matrix(train_predictions, keys),
            y_train,
            model=model,
            calibrate=calibrate,
            seed=seed,
        )

        def score_predictions(predictions: Sequence[Prediction]) -> np.ndarray:
            return fitted.predict_proba(_prediction_matrix(predictions, keys))[:, 1]

        return score_predictions

    return fit_predictions


def select_features_by_ci(
    base_features: Sequence[str],
    increment_ci95: Mapping[str, tuple[float, float]],
) -> list[str]:
    """Включить кандидата только при строго положительной нижней границе Δ."""
    selected = list(base_features)
    if len(selected) != len(set(selected)):
        raise ValueError("base_features contains duplicates")
    for feature, interval in increment_ci95.items():
        if len(interval) != 2:
            raise ValueError(f"CI for feature {feature!r} must contain two bounds")
        lower, upper = (float(interval[0]), float(interval[1]))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError(f"Invalid CI for feature {feature!r}: {interval!r}")
        if feature not in selected and lower > 0.0:
            selected.append(feature)
    return selected
