# tests/test_surface_oof.py
"""OOF-контур surface: главное свойство — обучение не видит оцениваемый кейс."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rag_reliability.methods.surface.oof import (
    Folds,
    evaluable_samples,
    fit_majority_head,
    load_folds,
    oof_probabilities,
)
from rag_reliability.schema import RagSample


def _samples(n: int) -> list[RagSample]:
    return [
        RagSample(
            id=f"case_{index:03d}",
            question=f"вопрос {index}",
            context=f"[CHUNK 1]\nконтекст {index} со значением {index * 10}",
            answer=f"ответ {index} со значением {index * 10}",
            faithfulness=index % 2,
            relevance=(index // 2) % 2,
        )
        for index in range(n)
    ]


def _folds(samples: list[RagSample], *, n_folds: int = 5, n_repeats: int = 2) -> Folds:
    assignment = {
        sample.id: [(index + repeat) % n_folds for repeat in range(n_repeats)]
        for index, sample in enumerate(samples)
    }
    return Folds(
        assignment=assignment,
        n_folds=n_folds,
        n_repeats=n_repeats,
        corpus_n=len(samples),
        sha256="0" * 64,
    )


def _features(samples: list[RagSample]) -> np.ndarray:
    return np.asarray([[float(index), float(index % 3)] for index, _ in enumerate(samples)])


def test_training_never_sees_the_scored_case() -> None:
    """Ядро задачи: кейс скорится моделью, обученной без него.

    Без этого свойства surf.p_faith — признак, видевший собственную метку, и
    прирост стэка C1 оказывается завышенным.
    """
    samples = _samples(40)
    folds = _folds(samples)
    features = _features(samples)
    seen_during_fit: list[set[float]] = []

    def spy_head(x: np.ndarray, y: np.ndarray, seed: int):  # noqa: ARG001
        seen_during_fit.append({float(row[0]) for row in x})
        return lambda xn: np.full(len(xn), 0.5)

    oof_probabilities(samples, features, folds, variant="surface", fit_head=spy_head)

    # Каждый фит должен был получить строго меньше корпуса: held-out фолд отсутствует.
    assert seen_during_fit
    assert all(len(seen) < len(samples) for seen in seen_during_fit)


def test_held_out_case_is_absent_from_its_own_training_fold() -> None:
    samples = _samples(25)
    folds = _folds(samples, n_repeats=1)
    features = _features(samples)
    leaks: list[str] = []

    def make_spy(train_rows: set[float]):
        def head(xn: np.ndarray) -> np.ndarray:
            for row in xn:
                if float(row[0]) in train_rows:
                    leaks.append(str(row[0]))
            return np.full(len(xn), 0.5)

        return head

    def spy_head(x: np.ndarray, y: np.ndarray, seed: int):  # noqa: ARG001
        return make_spy({float(row[0]) for row in x})

    oof_probabilities(samples, features, folds, variant="surface", fit_head=spy_head)

    assert leaks == []


def test_every_case_gets_exactly_one_averaged_probability() -> None:
    samples = _samples(30)
    result = oof_probabilities(samples, _features(samples), _folds(samples), variant="surface")

    assert set(result) == {sample.id for sample in samples}
    assert all(0.0 <= p_faith <= 1.0 and 0.0 <= p_rel <= 1.0 for p_faith, p_rel in result.values())


def test_majority_head_returns_the_train_base_rate() -> None:
    head = fit_majority_head(np.zeros((4, 2)), np.array([1, 1, 1, 0]), seed=0)
    assert head(np.zeros((3, 2))).tolist() == [0.75, 0.75, 0.75]


def test_majority_ignores_features_entirely() -> None:
    samples = _samples(20)
    folds = _folds(samples)
    plain = oof_probabilities(samples, _features(samples), folds, variant="majority")
    shuffled = oof_probabilities(samples, _features(samples)[::-1], folds, variant="majority")

    assert plain == shuffled


def test_single_class_fold_does_not_crash_the_logreg_head() -> None:
    """Вырожденный фолд — не повод потерять кейс: голова становится константой."""
    samples = [
        RagSample(id=f"c{i}", question="q", context="c", answer="a", faithfulness=1, relevance=1)
        for i in range(15)
    ]
    result = oof_probabilities(samples, _features(samples), _folds(samples), variant="surface")

    assert len(result) == 15
    assert all(p == 1.0 for p, _ in result.values())


def test_unassigned_sample_is_a_loud_error_not_a_default() -> None:
    samples = _samples(10)
    folds = _folds(samples)
    stranger = RagSample(
        id="stranger", question="q", context="c", answer="a", faithfulness=1, relevance=0
    )

    with pytest.raises(ValueError, match="no fold assignment"):
        oof_probabilities(
            [*samples, stranger], _features([*samples, stranger]), folds, variant="surface"
        )


def test_evaluable_samples_keeps_corpus_order_and_drops_unassigned() -> None:
    samples = _samples(6)
    folds = _folds(samples)
    trimmed = Folds(
        assignment={sample.id: folds.assignment[sample.id] for sample in samples[:4]},
        n_folds=folds.n_folds,
        n_repeats=folds.n_repeats,
        corpus_n=folds.corpus_n,
        sha256=folds.sha256,
    )

    kept = evaluable_samples(samples, trimmed)

    assert [sample.id for sample in kept] == [sample.id for sample in samples[:4]]


def test_unknown_variant_lists_the_available_ones() -> None:
    samples = _samples(5)
    with pytest.raises(ValueError, match="surface"):
        oof_probabilities(samples, _features(samples), _folds(samples), variant="nope")


def test_load_folds_rejects_a_short_assignment(tmp_path: Path) -> None:
    path = tmp_path / "folds.json"
    path.write_text(
        json.dumps(
            {
                "config": {"n_folds": 5, "n_repeats": 5},
                "corpus": {"n": 2, "sha256": "x"},
                "assignment": {"a": [0, 1, 2, 3, 4], "b": [0]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="shorter than n_repeats"):
        load_folds(path)


def test_load_real_folds_file_matches_its_declared_shape() -> None:
    folds = load_folds("data/splits/folds.json")

    assert folds.n_folds == 5
    assert folds.n_repeats == 5
    assert len(folds.assignment) < folds.corpus_n  # oversized-группы исключены
