"""Детектор пропущенных условий: то, что entailment невидит по построению.

Reference-free метрики faithfulness измеряют только **precision** — проверяется
сказанное, а не пропущенное. Уклончивый и неполный ответ получает высокий скор.
Классы ошибок ``reason_missed_chunk_conditions`` и ``reason_incomplete_answer``
структурно невидимы для grounding: каждое произнесённое предложение подкреплено,
а условие из чанка просто не произнесено.

Эвристика намеренно грубая (регулярки + числа), но ловит ровно тот класс.
NLI здесь не участвует: область покрытия задаётся чанками, которые grounding
пометил источниками опоры (``E.argmax``), поэтому своих NLI-вызовов нет.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

#: Ровно 4 ключа, объявленные карточкой C4 §3.
COVERAGE_KEYS: tuple[str, ...] = (
    "m6.n_conditions",
    "m6.cond_coverage",
    "m6.digit_coverage",
    "m6.uncovered_max",
)

#: Маркеры условий из карточки C4 §3.
CONDITION_PATTERNS: tuple[str, ...] = (
    r"\bесли\b",
    r"\bпри условии\b",
    r"\bне менее\b",
    r"\bне более\b",
    r"\bдо \d+",
    r"\bот \d+",
    r"\bв течение\b",
    r"\bне позднее\b",
    r"\bтолько для\b",
    r"\bкроме\b",
    r"\bминимальн",
    r"\bмаксимальн",
    r"\d+(?:[.,]\d+)?\s*%",
    r"\d+\s*(?:руб|₽|дн|мес|год|час)",
)

_CONDITION_RE = re.compile("|".join(CONDITION_PATTERNS), re.IGNORECASE | re.UNICODE)

#: Числа: разряды через пробел/неразрывный пробел («100 000») и десятичные («0,5»).
_NUMBER_RE = re.compile(r"\d{1,3}(?:[\s ]\d{3})+|\d+(?:[.,]\d+)?", re.UNICODE)

#: Сколько символов после маркера считать частью условия. «не менее 100 руб.» —
#: числовое ядро стоит справа от маркера, без окна условие осталось бы без числа.
LOOKAHEAD_CHARS = 30
#: Маркеры ближе этого расстояния описывают одно условие («не менее» + «100 руб»).
MERGE_GAP_CHARS = 8

#: Вес условия для ``uncovered_max``: числовое ядро проверяемо и потому весит
#: больше чисто лексического маркера, который мог попасть в текст случайно.
NUMERIC_WEIGHT = 1.0
LEXICAL_WEIGHT = 0.5


@dataclass(frozen=True)
class Condition:
    """Одно условие из чанка: текст маркера, его числовое ядро и вес."""

    text: str
    numbers: frozenset[str]
    weight: float


def normalize_numbers(text: str) -> set[str]:
    """Числа текста в канонической форме: «100 000» и «100000» — одно число."""
    numbers: set[str] = set()
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0).replace(" ", "").replace(" ", "").replace(",", ".")
        try:
            numbers.add(f"{float(raw):.6g}")
        except ValueError:  # pragma: no cover - регулярка не пропускает такое
            continue
    return numbers


def _merged_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _CONDITION_RE.finditer(text):
        start, end = match.span()
        if spans and start - spans[-1][1] <= MERGE_GAP_CHARS:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans


def find_conditions(text: str) -> list[Condition]:
    """Условия чанка. Соседние маркеры схлопываются в одно условие."""
    conditions: list[Condition] = []
    for start, end in _merged_spans(text):
        marker = text[start:end].strip()
        numbers = frozenset(normalize_numbers(text[start : end + LOOKAHEAD_CHARS]))
        conditions.append(
            Condition(
                text=marker,
                numbers=numbers,
                weight=NUMERIC_WEIGHT if numbers else LEXICAL_WEIGHT,
            )
        )
    return conditions


def _is_covered(condition: Condition, answer: str, answer_numbers: set[str]) -> bool:
    """Условие покрыто, если его числовое ядро или сам маркер есть в ответе."""
    if condition.numbers:
        return condition.numbers <= answer_numbers
    return condition.text.lower() in answer.lower()


def coverage_features(
    answer: str,
    chunks: Sequence[str],
    *,
    source_chunk_ids: set[int],
) -> dict[str, float]:
    """Доля условий из РЕЛЕВАНТНЫХ чанков, отражённых в ответе.

    ``source_chunk_ids`` — чанки, помеченные grounding как источники опоры
    (``E.argmax``). Считать по всем 5–8 чанкам бессмысленно: там много условий,
    не относящихся к вопросу, и метрика превращается в шум.

    Пустое множество источников — ошибка, а не полное покрытие: кейс без
    источников означает сбой grounding, и молча выдать «всё покрыто» — ровно та
    ошибка, из-за которой битая строка признаков выглядела идеальным кейсом.
    """
    if not source_chunk_ids:
        raise ValueError(
            "coverage needs a non-empty source_chunk_ids (grounding E.argmax); "
            "an empty source set is a grounding failure, not full coverage"
        )
    out_of_range = sorted(index for index in source_chunk_ids if not 0 <= index < len(chunks))
    if out_of_range:
        raise IndexError(
            f"source chunk id(s) {out_of_range} outside the {len(chunks)} chunk(s) provided"
        )

    source_text = "\n".join(chunks[index] for index in sorted(source_chunk_ids))
    conditions = find_conditions(source_text)
    answer_numbers = normalize_numbers(answer)

    uncovered = [
        condition
        for condition in conditions
        if not _is_covered(condition, answer, answer_numbers)
    ]
    chunk_numbers = normalize_numbers(source_text)
    covered_numbers = chunk_numbers & answer_numbers

    return {
        "m6.n_conditions": float(len(conditions)),
        # Нечего покрывать — это не «всё пропущено»: кейс без условий не должен
        # выглядеть хуже кейса, где все условия отражены.
        "m6.cond_coverage": (
            1.0 if not conditions else float(len(conditions) - len(uncovered)) / len(conditions)
        ),
        "m6.digit_coverage": (
            1.0 if not chunk_numbers else float(len(covered_numbers)) / len(chunk_numbers)
        ),
        "m6.uncovered_max": (
            0.0 if not uncovered else float(max(condition.weight for condition in uncovered))
        ),
    }
