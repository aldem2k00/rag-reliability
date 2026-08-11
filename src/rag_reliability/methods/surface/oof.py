"""Out-of-fold вероятности surface/majority по data/splits/folds.json.

Почему не один фит на всём корпусе. `surf.p_faith`/`surf.p_rel` — выходы
логрега, и если обучить его на всех кейсах, а потом ими же и скорить, стэкер
C1 получит признак, видевший собственную метку. Прирост стэка окажется
завышенным ровно тем способом, ради устранения которого затевалась фаза 0.
Поэтому каждый кейс скорится моделью, обученной без него.

Кейсы без номера фолда здесь отсутствуют: `folds.json` исключает oversized-группы,
и предсказать их out-of-fold физически нечем.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rag_reliability.schema import RagSample

VARIANTS: tuple[str, ...] = ("surface", "majority")


@dataclass(frozen=True)
class Folds:
    """Разбиение из folds.json: assignment[id][repeat] -> номер фолда."""

    assignment: dict[str, list[int]]
    n_folds: int
    n_repeats: int
    corpus_n: int
    sha256: str

    def repeats(self) -> range:
        return range(self.n_repeats)


def load_folds(path: str | Path) -> Folds:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("config", "corpus", "assignment"):
        if key not in payload:
            raise ValueError(f"{path}: folds.json has no {key!r} section")
    config, corpus = payload["config"], payload["corpus"]
    assignment = {str(key): list(value) for key, value in payload["assignment"].items()}

    n_repeats = int(config["n_repeats"])
    wrong = [key for key, value in assignment.items() if len(value) != n_repeats]
    if wrong:
        raise ValueError(
            f"{path}: {len(wrong)} id(s) have an assignment shorter than "
            f"n_repeats={n_repeats}: {wrong[:5]}"
        )
    return Folds(
        assignment=assignment,
        n_folds=int(config["n_folds"]),
        n_repeats=n_repeats,
        corpus_n=int(corpus["n"]),
        sha256=str(corpus["sha256"]),
    )


def evaluable_samples(samples: Sequence[RagSample], folds: Folds) -> list[RagSample]:
    """Кейсы, которым folds.json назначил фолд. Порядок корпуса сохраняется."""
    return [sample for sample in samples if sample.id in folds.assignment]


Head = Callable[[np.ndarray], np.ndarray]
FitHead = Callable[[np.ndarray, np.ndarray, int], Head]


def fit_logreg_head(x: np.ndarray, y: np.ndarray, seed: int) -> Head:
    """Одна ось (faith или rel): scaler + logreg; при одном классе — константа."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    classes = set(y.tolist())
    if len(classes) < 2:
        constant = float(next(iter(classes))) if classes else 0.5
        return lambda xn: np.full(len(xn), constant)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(scaler.transform(x), y)
    return lambda xn: model.predict_proba(scaler.transform(xn))[:, 1]


def fit_majority_head(x: np.ndarray, y: np.ndarray, seed: int) -> Head:  # noqa: ARG001
    """Базовая ставка train-части: константа, не зависящая от фич."""
    rate = float(np.mean(y)) if len(y) else 0.5
    return lambda xn: np.full(len(xn), rate)


_HEADS: dict[str, FitHead] = {"surface": fit_logreg_head, "majority": fit_majority_head}


def oof_probabilities(
    samples: Sequence[RagSample],
    features: np.ndarray,
    folds: Folds,
    *,
    variant: str,
    seed: int = 0,
    fit_head: FitHead | None = None,
) -> dict[str, tuple[float, float]]:
    """OOF-вероятности (p_faith, p_rel), усреднённые по повторам folds.json.

    Усреднение по 5 повторам не нарушает честность: в каждом повторе кейс
    предсказывается моделью, не видевшей его метку, — а разброс единичного
    разбиения (sd 0.032) заметно больше разницы между методами.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown surface variant {variant!r}; available: {list(VARIANTS)}")
    head_factory = fit_head or _HEADS[variant]

    ids = [sample.id for sample in samples]
    missing = [sample_id for sample_id in ids if sample_id not in folds.assignment]
    if missing:
        raise ValueError(
            f"{len(missing)} sample(s) have no fold assignment: {missing[:5]}. "
            "Filter the corpus with evaluable_samples() first."
        )
    if features.shape[0] != len(ids):
        raise ValueError(
            f"Feature matrix has {features.shape[0]} row(s) for {len(ids)} sample(s)"
        )

    y_faith = np.asarray([sample.faithfulness for sample in samples], dtype=int)
    y_rel = np.asarray([sample.relevance for sample in samples], dtype=int)
    index = {sample_id: position for position, sample_id in enumerate(ids)}

    totals = np.zeros((len(ids), 2), dtype=float)
    counts = np.zeros(len(ids), dtype=int)

    for repeat in folds.repeats():
        fold_of = np.asarray([folds.assignment[sample_id][repeat] for sample_id in ids])
        for fold in range(folds.n_folds):
            test_mask = fold_of == fold
            train_mask = ~test_mask
            if not test_mask.any() or not train_mask.any():
                continue
            head_f = head_factory(features[train_mask], y_faith[train_mask], seed)
            head_r = head_factory(features[train_mask], y_rel[train_mask], seed)
            totals[test_mask, 0] += head_f(features[test_mask])
            totals[test_mask, 1] += head_r(features[test_mask])
            counts[test_mask] += 1

    never_scored = [ids[position] for position, count in enumerate(counts) if count == 0]
    if never_scored:
        raise ValueError(
            f"{len(never_scored)} sample(s) never landed in a held-out fold: "
            f"{never_scored[:5]}"
        )
    averaged = totals / counts[:, None]
    return {
        sample_id: (float(averaged[index[sample_id], 0]), float(averaged[index[sample_id], 1]))
        for sample_id in ids
    }


def corpus_sha256(path: str | Path) -> str:
    """Хэш файла корпуса — чтобы артефакт нельзя было спутать с другим сплитом."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
