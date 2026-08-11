"""Поверхностные фичи кейса: длины, n-грамм overlap, совпадение чисел, навигация.

Перенос из ветки `m3-m6` (`baselines/surface.py`). Формулы сохранены дословно —
иначе числа новой ветки нельзя сравнивать с опубликованными. Единственное
содержательное отличие: там контекст приходил списком чанков (`Case.context`),
здесь `RagSample.context` — строка, поэтому число чанков восстанавливается по
маркерам `[CHUNK n]`, которыми корпус организаторов размечен.

Функции чистые: ни модели, ни обучения, ни утечки. Обучаемая голова живёт
в `oof.py` и фитится строго внутри train-части фолда.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_reliability.schema import RagSample

_WORD_RE = re.compile(r"\w+")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_CHUNK_RE = re.compile(r"\[CHUNK\s+\d+\]")
# URL/навигационные паттерны — прокси для маркеров wrong_navigation/answer_for_operator
_NAV_RE = re.compile(
    r"https?://|www\.|в\s+приложени\w*|в\s+раздел\w*|по\s+ссылк\w*|стать[еия]\w*",
    re.IGNORECASE,
)

# Фиксированный порядок фич для матрицы (детерминизм).
FEATURE_KEYS: tuple[str, ...] = (
    "len_answer",
    "len_ctx",
    "len_query",
    "n_chunks",
    "overlap_ans_ctx_1",
    "overlap_ans_ctx_2",
    "overlap_ans_q_1",
    "digit_match_ratio",
    "url_or_nav_ratio",
)


def _tokens(text: str) -> list[str]:
    """Токенизация: lower + все \\w+ последовательности."""
    return _WORD_RE.findall(text.lower())


def split_chunks(context: str) -> list[str]:
    """Разбить контекст по маркерам `[CHUNK n]`.

    Без маркеров весь контекст считается одним чанком: пустой список сделал бы
    фичу n_chunks нулевой и неотличимой от «контекста нет».
    """
    parts = [part.strip() for part in _CHUNK_RE.split(context) if part.strip()]
    return parts if parts else ([context] if context.strip() else [])


def ngram_overlap(a: str, b: str, n: int = 2) -> float:
    """Доля словных n-грамм строки a, встречающихся среди n-грамм строки b.

    Если в a меньше n слов — откат к n=1; пустая a → 0.0. Результат в [0, 1].
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta:
        return 0.0
    if len(ta) < n:
        n = 1
    a_ngrams = [tuple(ta[i : i + n]) for i in range(len(ta) - n + 1)]
    b_ngrams = {tuple(tb[i : i + n]) for i in range(len(tb) - n + 1)} if len(tb) >= n else set()
    hits = sum(1 for gram in a_ngrams if gram in b_ngrams)
    return hits / max(1, len(a_ngrams))


def _norm_num(value: str) -> str:
    """Нормализация числа как строки: запятая → точка, без хвостовых нулей дроби."""
    value = value.replace(",", ".")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def surface_features(sample: RagSample) -> dict[str, float]:
    """Поверхностные фичи кейса в фиксированном наборе FEATURE_KEYS."""
    answer_tokens = _tokens(sample.answer)
    question_tokens = _tokens(sample.question)
    context_tokens = _tokens(sample.context)

    answer_numbers = [_norm_num(m) for m in _NUM_RE.findall(sample.answer)]
    context_numbers = {_norm_num(m) for m in _NUM_RE.findall(sample.context)}
    if answer_numbers:
        digit_ratio = sum(1 for x in answer_numbers if x in context_numbers) / len(answer_numbers)
    else:
        digit_ratio = 1.0  # нет чисел в ответе — нечему противоречить

    nav_hits = len(_NAV_RE.findall(sample.answer))
    return {
        "len_answer": float(len(answer_tokens)),
        "len_ctx": float(len(context_tokens)),
        "len_query": float(len(question_tokens)),
        "n_chunks": float(len(split_chunks(sample.context))),
        "overlap_ans_ctx_1": ngram_overlap(sample.answer, sample.context, n=1),
        "overlap_ans_ctx_2": ngram_overlap(sample.answer, sample.context, n=2),
        "overlap_ans_q_1": ngram_overlap(sample.answer, sample.question, n=1),
        "digit_match_ratio": float(digit_ratio),
        "url_or_nav_ratio": nav_hits / max(1, len(answer_tokens)),
    }


def feature_vector(sample: RagSample) -> list[float]:
    """Фичи в порядке FEATURE_KEYS — то, что уходит в матрицу обучения."""
    features = surface_features(sample)
    return [features[key] for key in FEATURE_KEYS]
