"""Текстовый вход энкодера: формат, разделители, усечение без потери ответа.

Токенизатор — заглушка на пробелах: тест обязан идти без torch, transformers и
без единого скачанного веса.
"""

from __future__ import annotations

import pytest

from rag_reliability.methods.encoder.data import (
    ANSWER_HEADER,
    DIALOG_HEADER,
    QUESTION_HEADER,
    build_encoder_text,
    encode,
    make_examples,
    parse_chunks,
    split_dialog,
)
from rag_reliability.schema import RagSample

DIALOG = (
    "Ассистент: Приветствую! На связи Альфа-Помощник\n"
    "Клиент: не приходит смс\n"
    "Ассистент: Уточните номер телефона\n"
    "Клиент: Как подключить Alfa Pay?"
)
CONTEXT = "[CHUNK 1]\nПодключите Alfa Pay в настройках карты.\n\n[CHUNK 2]\nЛимит операции 3000 рублей."
ANSWER = "Откройте карту и выберите оплату смартфоном."


class WordTokenizer:
    """Пробельная токенизация: id — позиция слова, спецтокены — два фиктивных."""

    n_special = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [len(word) for word in text.split()]
        return self.build_inputs_with_special_tokens(ids) if add_special_tokens else ids

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return self.n_special

    def build_inputs_with_special_tokens(self, token_ids_0: list[int]) -> list[int]:
        return [-1, *token_ids_0, -2]


def make_sample(
    question: str = DIALOG,
    context: str = CONTEXT,
    answer: str = ANSWER,
    faithfulness: int = 1,
    relevance: int = 1,
    sample_id: str = "organizer_000001",
) -> RagSample:
    return RagSample(
        id=sample_id,
        question=question,
        context=context,
        answer=answer,
        faithfulness=faithfulness,
        relevance=relevance,
        marker="none" if faithfulness and relevance else "unknown",
    )


# --------------------------------------------------------------------------- #
# Разбор диалога и чанков
# --------------------------------------------------------------------------- #


def test_split_dialog_separates_last_client_turn_from_history() -> None:
    history, current = split_dialog(DIALOG)

    assert current == "Как подключить Alfa Pay?"
    assert current not in history
    assert "не приходит смс" in history


def test_split_dialog_keeps_turns_that_follow_the_last_client_turn() -> None:
    dialog = "Клиент: где карта\nАссистент: карта в пути"

    history, current = split_dialog(dialog)

    assert current == "где карта"
    assert "Ассистент: карта в пути" in history


def test_split_dialog_without_client_turn_keeps_everything_as_history() -> None:
    dialog = "Ассистент: Приветствую!"

    history, current = split_dialog(dialog)

    assert history == dialog
    assert current == ""


def test_parse_chunks_splits_on_markers() -> None:
    assert parse_chunks(CONTEXT) == [
        "Подключите Alfa Pay в настройках карты.",
        "Лимит операции 3000 рублей.",
    ]


def test_parse_chunks_treats_unmarked_context_as_a_single_chunk() -> None:
    assert parse_chunks("сплошной текст") == ["сплошной текст"]


def test_parse_chunks_of_empty_context_is_empty() -> None:
    assert parse_chunks("   ") == []


# --------------------------------------------------------------------------- #
# Формат входа
# --------------------------------------------------------------------------- #


def test_encoder_text_marks_chunk_boundaries_in_russian() -> None:
    text = build_encoder_text(make_sample())

    assert "[Чанк 1]" in text
    assert "[Чанк 2]" in text
    assert "[CHUNK" not in text


def test_encoder_text_separates_dialog_history_from_current_question() -> None:
    text = build_encoder_text(make_sample())

    assert DIALOG_HEADER in text
    assert f"{QUESTION_HEADER} Как подключить Alfa Pay?" in text
    assert text.index(DIALOG_HEADER) < text.index(QUESTION_HEADER)


def test_encoder_text_puts_the_answer_last() -> None:
    text = build_encoder_text(make_sample())

    assert text.index("[Чанк 1]") < text.index(ANSWER_HEADER)
    assert text.rstrip().endswith(ANSWER)


def test_encoder_text_without_context_still_has_dialog_and_answer() -> None:
    text = build_encoder_text(make_sample(context=""))

    assert "[Чанк" not in text
    assert text.rstrip().endswith(ANSWER)


# --------------------------------------------------------------------------- #
# Усечение
# --------------------------------------------------------------------------- #


def test_encode_keeps_everything_when_the_budget_is_generous() -> None:
    tokenizer = WordTokenizer()
    sample = make_sample()

    encoded = encode(sample, tokenizer, max_length=1000)

    assert encoded.truncated is False
    assert encoded.n_chunks_kept == encoded.n_chunks_total == 2
    assert encoded.input_ids[0] == -1
    assert encoded.input_ids[-1] == -2
    assert encoded.attention_mask == [1] * len(encoded.input_ids)


def test_encode_never_exceeds_max_length() -> None:
    tokenizer = WordTokenizer()
    sample = make_sample(context="[CHUNK 1]\n" + "слово " * 400)

    for max_length in (8, 16, 32, 64, 128):
        encoded = encode(sample, tokenizer, max_length=max_length)
        assert len(encoded.input_ids) <= max_length, max_length


def test_encode_drops_context_before_touching_the_answer() -> None:
    """Ответ идёт последним, поэтому обычная хвостовая обрезка съела бы именно его."""
    tokenizer = WordTokenizer()
    sample = make_sample(context="[CHUNK 1]\n" + "контекст " * 500)
    answer_ids = tokenizer.encode("\n\n" + f"{ANSWER_HEADER}\n{ANSWER}", add_special_tokens=False)

    encoded = encode(sample, tokenizer, max_length=64)

    assert encoded.truncated is True
    assert encoded.n_chunks_total == 1
    # Хвост уцелел целиком и стоит перед закрывающим спецтокеном.
    assert encoded.input_ids[-1 - len(answer_ids) : -1] == answer_ids


def test_encode_keeps_the_answer_when_the_dialog_alone_overflows() -> None:
    tokenizer = WordTokenizer()
    sample = make_sample(question="Клиент: " + "вопрос " * 500, context="")
    answer_ids = tokenizer.encode("\n\n" + f"{ANSWER_HEADER}\n{ANSWER}", add_special_tokens=False)

    encoded = encode(sample, tokenizer, max_length=32)

    assert len(encoded.input_ids) <= 32
    assert encoded.input_ids[-1 - len(answer_ids) : -1] == answer_ids


def test_encode_truncates_the_answer_only_when_it_alone_overflows() -> None:
    tokenizer = WordTokenizer()
    sample = make_sample(answer="ответ " * 500)

    encoded = encode(sample, tokenizer, max_length=16)

    assert encoded.truncated is True
    assert encoded.n_chunks_kept == 0
    assert len(encoded.input_ids) == 16


def test_encode_keeps_more_chunks_as_the_budget_grows() -> None:
    """Монотонность: длинное окно не может видеть меньше источников, чем короткое."""
    tokenizer = WordTokenizer()
    sample = make_sample(
        context="".join(f"[CHUNK {i}]\n" + "слово " * 20 + "\n\n" for i in range(1, 9))
    )

    kept = [encode(sample, tokenizer, max_length=length).n_chunks_kept for length in (64, 128, 512)]

    assert kept == sorted(kept)
    assert kept[-1] == 8


def test_encode_rejects_a_window_that_cannot_hold_special_tokens() -> None:
    tokenizer = WordTokenizer()

    with pytest.raises(ValueError, match="special token"):
        encode(make_sample(), tokenizer, max_length=WordTokenizer.n_special)


def test_encode_rejects_a_non_positive_window() -> None:
    with pytest.raises(ValueError, match="max_length must be >= 1"):
        encode(make_sample(), WordTokenizer(), max_length=0)


# --------------------------------------------------------------------------- #
# Примеры для обучения
# --------------------------------------------------------------------------- #


def test_make_examples_labels_reliable_as_conjunction_of_both_axes() -> None:
    tokenizer = WordTokenizer()
    samples = [
        make_sample(faithfulness=1, relevance=1, sample_id="a"),
        make_sample(faithfulness=1, relevance=0, sample_id="b"),
        make_sample(faithfulness=0, relevance=1, sample_id="c"),
        make_sample(faithfulness=0, relevance=0, sample_id="d"),
    ]

    examples = make_examples(samples, tokenizer, max_length=256)

    assert [example.label for example in examples] == [1, 0, 0, 0]
    assert [example.id for example in examples] == ["a", "b", "c", "d"]


def test_make_examples_reports_truncation_per_case() -> None:
    tokenizer = WordTokenizer()
    samples = [
        make_sample(sample_id="short"),
        make_sample(sample_id="long", context="[CHUNK 1]\n" + "слово " * 500),
    ]

    examples = make_examples(samples, tokenizer, max_length=128)

    assert [example.truncated for example in examples] == [False, True]
