"""Текстовый вход энкодера и посегментное усечение под ``max_length``.

Формат входа взят из ветки ``m3-m6`` (спека ``20_PHASE1`` §3.3д), а не из
``prepare_data.py``: там весь диалог лежит в одном поле, а чанки помечены
латинским ``[CHUNK i]``. Явные границы источников обязательны — ``chunk_fact_mixup``
один из двух главных типов ошибок корпуса, и без разделителей модель не может
отличить, из какого чанка пришёл факт.

Усечение посегментное, а не хвостовое: ответ бота стоит последним и при
``max_length`` меньше длины входа обычным ``truncation=True`` он бы выпадал
первым — модель училась бы предсказывать надёжность ответа, которого не видела.

Модуль чистый: ни torch, ни transformers. Токенизатор передаётся объектом с
HF-совместимыми ``encode``/``num_special_tokens_to_add``/``build_inputs_with_special_tokens``,
поэтому тесты обходятся заглушкой и не грузят веса.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rag_reliability.schema import RagSample

#: Роли, которыми размечен диалог корпуса организаторов (тот же набор, что в
#: ``splits.py``: разъехавшись, они дали бы разные «последние реплики клиента»
#: у группировки и у обучения).
ROLE_LINE = re.compile(r"(?m)^(Клиент|Ассистент|Оператор|AlfaGen)\s*:[ \t]*")
CLIENT_ROLE = "Клиент"

#: Маркер чанка в сыром корпусе. На выходе он переписывается в ``[Чанк i]``.
CHUNK_MARKER = re.compile(r"\[CHUNK\s+\d+\]")

DIALOG_HEADER = "История диалога:"
QUESTION_HEADER = "Текущий вопрос:"
ANSWER_HEADER = "Ответ бота:"


class Tokenizer(Protocol):
    """Минимум HF-API, который нужен построению входа."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def num_special_tokens_to_add(self, pair: bool = ...) -> int: ...

    def build_inputs_with_special_tokens(self, token_ids_0: list[int]) -> list[int]: ...


@dataclass(frozen=True)
class EncoderSegments:
    """Вход, разложенный на части с разными приоритетами при усечении."""

    head: str
    chunks: tuple[str, ...]
    tail: str


@dataclass(frozen=True)
class EncodedInput:
    """Готовый вход модели плюс честный след того, что пришлось выбросить."""

    input_ids: list[int]
    n_chunks_kept: int
    n_chunks_total: int
    truncated: bool

    @property
    def attention_mask(self) -> list[int]:
        return [1] * len(self.input_ids)


@dataclass(frozen=True)
class EncoderExample:
    """Один обучающий пример: id нужен, чтобы OOF-логит вернулся своему кейсу."""

    id: str
    input_ids: list[int]
    label: int
    truncated: bool


def split_dialog(question: str) -> tuple[str, str]:
    """Разделить диалог на историю и последнюю реплику клиента.

    История — диалог *без* этой реплики, а не префикс до неё: в части кейсов
    после вопроса клиента идёт ещё реплика ассистента, и обрезкой по префиксу
    она бы потерялась.

    Реплики клиента нет вовсе (15 строк корпуса) — весь текст уходит в историю,
    текущий вопрос пуст. Заголовок при этом сохраняется: одинаковая структура
    входа важнее, чем экономия одной строки на 0.7% кейсов.
    """
    roles = list(ROLE_LINE.finditer(question))
    for index in range(len(roles) - 1, -1, -1):
        match = roles[index]
        if match.group(1) != CLIENT_ROLE:
            continue
        end = roles[index + 1].start() if index + 1 < len(roles) else len(question)
        current = question[match.end() : end].strip()
        history = (question[: match.start()].rstrip() + "\n" + question[end:].lstrip()).strip()
        return history, current
    return question.strip(), ""


def parse_chunks(context: str) -> list[str]:
    """Разбить контекст по маркерам ``[CHUNK n]``.

    Без маркеров весь контекст считается одним чанком — так же, как в
    ``surface/features.py``: пустой список означал бы «контекста нет», а это
    другой случай.
    """
    parts = [part.strip() for part in CHUNK_MARKER.split(context) if part.strip()]
    if parts:
        return parts
    return [context.strip()] if context.strip() else []


def build_segments(sample: RagSample) -> EncoderSegments:
    """Разложить кейс на голову (диалог + вопрос), чанки и хвост (ответ)."""
    history, current = split_dialog(sample.question)
    head = f"{DIALOG_HEADER}\n{history}\n\n{QUESTION_HEADER} {current}".rstrip()
    chunks = tuple(
        f"[Чанк {index}]\n{chunk}" for index, chunk in enumerate(parse_chunks(sample.context), 1)
    )
    tail = f"{ANSWER_HEADER}\n{sample.answer}"
    return EncoderSegments(head=head, chunks=chunks, tail=tail)


def build_encoder_text(sample: RagSample) -> str:
    """Полный текстовый вход энкодера без усечения."""
    return render_segments(build_segments(sample))


def render_segments(segments: EncoderSegments) -> str:
    blocks = [segments.head]
    if segments.chunks:
        blocks.append("\n".join(segments.chunks))
    blocks.append(segments.tail)
    return "\n\n".join(blocks)


def _encode_plain(tokenizer: Tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def encode(sample: RagSample, tokenizer: Tokenizer, max_length: int) -> EncodedInput:
    """Токенизировать кейс, укладываясь в ``max_length`` с сохранением ответа.

    Порядок жертв при переполнении: сначала чанки (с конца), затем голова,
    и только если ответ один длиннее всего окна — сам ответ. Ответ бота —
    предмет предсказания, и вход без него бессмысленен.

    Границы сегментов токенизируются по отдельности, поэтому результат может
    отличаться на единицы токенов от токенизации склеенного текста. Это цена
    контролируемого усечения и она заведомо меньше, чем потеря ответа.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")
    n_special = tokenizer.num_special_tokens_to_add()
    budget = max_length - n_special
    if budget < 1:
        raise ValueError(
            f"max_length={max_length} leaves no room for content: the tokenizer adds "
            f"{n_special} special token(s)"
        )

    segments = build_segments(sample)
    head_ids = _encode_plain(tokenizer, segments.head + "\n\n")
    tail_ids = _encode_plain(tokenizer, "\n\n" + segments.tail)
    truncated = False

    if len(tail_ids) >= budget:
        # Ответ один не влезает в окно: голова и чанки отбрасываются целиком,
        # ответ режется с конца. Хвостовая обрезка тут неизбежна, но заметна.
        return EncodedInput(
            input_ids=tokenizer.build_inputs_with_special_tokens(tail_ids[:budget]),
            n_chunks_kept=0,
            n_chunks_total=len(segments.chunks),
            truncated=True,
        )

    if len(head_ids) + len(tail_ids) > budget:
        head_ids = head_ids[: budget - len(tail_ids)]
        truncated = True

    room = budget - len(head_ids) - len(tail_ids)
    kept: list[int] = []
    n_chunks_kept = 0
    for chunk in segments.chunks:
        if room <= 0:
            truncated = True
            break
        chunk_ids = _encode_plain(tokenizer, chunk if n_chunks_kept == 0 else "\n" + chunk)
        if len(chunk_ids) > room:
            kept.extend(chunk_ids[:room])
            n_chunks_kept += 1
            room = 0
            truncated = True
            break
        kept.extend(chunk_ids)
        n_chunks_kept += 1
        room -= len(chunk_ids)

    return EncodedInput(
        input_ids=tokenizer.build_inputs_with_special_tokens(head_ids + kept + tail_ids),
        n_chunks_kept=n_chunks_kept,
        n_chunks_total=len(segments.chunks),
        truncated=truncated,
    )


def make_examples(
    samples: Sequence[RagSample], tokenizer: Tokenizer, max_length: int
) -> list[EncoderExample]:
    """Обучающие примеры в порядке корпуса; метка — ``reliable = faith ∧ rel``."""
    examples: list[EncoderExample] = []
    for sample in samples:
        encoded = encode(sample, tokenizer, max_length)
        examples.append(
            EncoderExample(
                id=sample.id,
                input_ids=encoded.input_ids,
                label=sample.reliable,
                truncated=encoded.truncated,
            )
        )
    return examples
