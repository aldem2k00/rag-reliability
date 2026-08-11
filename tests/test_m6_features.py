"""Сентенизация Метода 6 на реальном примере из корпуса.

Регулярка ``[^.!?\\n]+[.!?]?|[^\\s]+`` резала банковский текст по каждой точке:
«макс.», «т.д.», «0.5%» и нумерация шагов давали фрагменты вместо предложений.
Для grounding это критично: гипотеза NLI — предложение, и обрывок «1.» не может
быть подкреплён никаким чанком, зато уверенно тянет ``min_entail`` вниз.

Тесты не требуют GPU и весов; razdel — чистый Python.
"""

from __future__ import annotations

import re

import pytest

razdel = pytest.importorskip("razdel", reason="razdel нужен для сентенизации Метода 6")

from rag_reliability.methods.m6.features import razdel_sentences, sentences  # noqa: E402

#: Ответ organizer_001543 из data/organizers.jsonl — пример с «руб.» и числами.
CORPUS_ANSWER = (
    "Максимальная сумма кэшбэка по категориям месяца — [NUMBER] руб. "
    "(для непремиальных карт). Партнёрский кэшбэк начисляется сверх этого лимита: "
    "например, у М.Видео. Эльдорадо лимит — [NUMBER] руб., у Дикси Доставка — "
    "[NUMBER] руб. Таким образом, если вы получите [NUMBER] руб. по категориям и "
    "[NUMBER] руб. по партнёрским предложениям, суммарно вы получите [NUMBER] руб. "
    "([NUMBER] руб. категорий + [NUMBER] руб. партнёрского кэшбэка)."
)

_LEGACY_RE = re.compile(r"[^.!?\n]+[.!?]?|[^\s]+", re.UNICODE)


def legacy_sentences(text: str) -> list[str]:
    """Прежняя регулярка — эталон «как не надо», а не рабочий код."""
    return [match.group(0).strip() for match in _LEGACY_RE.finditer(text) if match.group(0).strip()]


def test_corpus_answer_is_not_shredded() -> None:
    """12 «предложений» у регулярки против 4 настоящих у razdel."""
    assert len(legacy_sentences(CORPUS_ANSWER)) == 12

    result = razdel_sentences(CORPUS_ANSWER)

    assert len(result) == 4
    assert "".join(result.copy()) != ""
    # ни один фрагмент не является голым «руб.» или числом в скобках
    assert all(len(fragment.split()) > 2 for fragment in result)


def test_abbreviation_and_numbers_stay_inside_sentences() -> None:
    assert razdel_sentences("Ставка 0.5% годовых.") == ["Ставка 0.5% годовых."]
    assert razdel_sentences("Комиссии, переводы и т.д. не учитываются.") == [
        "Комиссии, переводы и т.д. не учитываются."
    ]


def test_numbered_steps_are_five_steps_not_twelve_fragments() -> None:
    instruction = (
        "1. Откройте приложение. 2. Выберите карту. 3. Нажмите «Оплатить». "
        "4. Подтвердите операцию. 5. Дождитесь чека."
    )

    result = razdel_sentences(instruction)

    assert len(result) == 5
    assert len(legacy_sentences(instruction)) == 10
    assert all(step[0].isdigit() for step in result)


def test_never_splits_more_than_the_legacy_regex() -> None:
    """Свойство, а не пример: razdel нигде не дробит текст сильнее регулярки."""
    texts = [
        CORPUS_ANSWER,
        "Лимит макс. 5000 руб. в месяц.",
        "Перевод до 100 000 ₽ в сутки без комиссии.",
        "Срок — не более 5 раб. дн. с момента обращения.",
        "Ответ.",
    ]
    for text in texts:
        assert len(razdel_sentences(text)) <= len(legacy_sentences(text))


def test_blank_text_keeps_a_single_fallback_span() -> None:
    """Кейс не может потерять гипотезу: пустой ответ — это одно «предложение»."""
    assert razdel_sentences("   ") == ["   "]
    assert razdel_sentences("") == [""]


def test_sentences_is_the_razdel_implementation() -> None:
    """Публичное имя не должно разъехаться с реализацией: багом было именно это."""
    assert sentences is razdel_sentences
