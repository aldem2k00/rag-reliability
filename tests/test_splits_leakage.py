"""Цена утечки: сколько получает тривиальный запоминатель под каждым протоколом.

Это не тест кода, а зафиксированное измерение, которое идёт в статью. 1-NN на
char-TF-IDF от (диалог + ответ) не умеет ничего, кроме поиска похожей строки в
train. Если под протоколом он обгоняет методы проекта, метрика меряет
запоминание, а не обобщение — это и есть обоснование group-aware разбиения.

Тест медленный (полный корпус, две TF-IDF матрицы) и потому выключен по
умолчанию: маркер ``slow`` плюс явное включение через ``RUN_SLOW_TESTS=1``::

    RUN_SLOW_TESTS=1 .venv/bin/python -m pytest tests/test_splits_leakage.py -q

Исключить его из быстрого прогона правкой ``Makefile``/``pyproject.toml`` нельзя:
оба файла принадлежат задаче A5. См. раздел «Требуется от других» в PR.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rag_reliability.dataset import load_jsonl
from rag_reliability.schema import RagSample

CORPUS = Path("data/organizers.jsonl")
FOLDS = Path("data/splits/folds.json")

#: Порог из карточки A1: под стратифицированным сплитом запоминатель обгоняет
#: лучший метод проекта (0.5982).
STRATIFIED_FLOOR = 0.60

#: Под group-aware разбиением ему остаётся заметно меньше.
GROUP_AWARE_CEILING = 0.55

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("RUN_SLOW_TESTS"),
        reason="медленный тест на полном корпусе; включается RUN_SLOW_TESTS=1",
    ),
]


def _memorizer_text(sample: RagSample) -> str:
    """То, что видит запоминатель: диалог и ответ бота, без контекста."""
    return f"{sample.question}\n{sample.answer}"


def _nn_macro_f1(train: list[RagSample], test: list[RagSample]) -> float:
    """macro-F1 по reliable для 1-NN на char 3-5 gram TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import f1_score
    from sklearn.neighbors import KNeighborsClassifier

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    train_matrix = vectorizer.fit_transform([_memorizer_text(s) for s in train])
    test_matrix = vectorizer.transform([_memorizer_text(s) for s in test])

    model = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    model.fit(train_matrix, [s.reliable for s in train])
    return float(f1_score([s.reliable for s in test], model.predict(test_matrix), average="macro"))


def nn_macro_f1_stratified() -> float:
    """Протокол A: стратифицированный split_samples, seed 42.

    Единственное разрешённое место вызова ``split_samples`` в новом коде
    (HANDOFF П4): регресс-тест, воспроизводящий исторические числа.
    """
    from rag_reliability.dataset import split_samples

    train, _, test = split_samples(load_jsonl(CORPUS), seed=42)
    return _nn_macro_f1(train, test)


def nn_macro_f1_group_aware(repeat: int = 0) -> float:
    """Протокол B: среднее по фолдам одного повтора канонического folds.json.

    Усреднение по фолдам, а не один фолд: при n_test ~ 297 одиночный фолд даёт
    0.52-0.58 в зависимости от того, какой именно взят, — это тот самый разброс
    единичного holdout, ради устранения которого фаза 0 и существует. Число,
    которое идёт в статью, должно быть кросс-валидационным.

    Кейсы oversized-групп отсутствуют в ``assignment``, но остаются в train во
    всех повторах — так их описывает карточка: исключены из метрики, не из
    обучения.
    """
    samples = load_jsonl(CORPUS)
    assignment = json.loads(FOLDS.read_text(encoding="utf-8"))["assignment"]
    n_folds = max(fold for folds in assignment.values() for fold in folds) + 1

    scores = []
    for fold in range(n_folds):
        test = [s for s in samples if s.id in assignment and assignment[s.id][repeat] == fold]
        train = [s for s in samples if s.id not in assignment or assignment[s.id][repeat] != fold]
        scores.append(_nn_macro_f1(train, test))
    return sum(scores) / len(scores)


@pytest.mark.skipif(not CORPUS.exists() or not FOLDS.exists(), reason="нет корпуса или фолдов")
def test_memorizer_gap_between_protocols() -> None:
    """Стратифицированный сплит даёт запоминателю ~0.63, group-aware ~0.51."""
    stratified = nn_macro_f1_stratified()
    group_aware = nn_macro_f1_group_aware()

    print(f"\n1-NN macro-F1: стратифицированный {stratified:.4f}, group-aware {group_aware:.4f}")
    assert stratified > STRATIFIED_FLOOR
    assert group_aware < GROUP_AWARE_CEILING
