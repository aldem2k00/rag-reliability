"""Обучающие примеры судьи: тот же формат, что и на инференсе.

Формат берётся из ``methods/m3/axes.py``, а не пишется здесь заново. Это не
экономия строк, а единственный способ получить симметрию: судья, обученный на
одном шаблоне и опрошенный другим, теряет якорь вердикта, и вероятность
приходит уже не из логпробов, а из regex-ветки — деградация, которая выглядит
как успешный прогон. ``check_format_symmetry`` проверяет это явно, до загрузки
модели.

Один кейс даёт два примера — по одному на ось. Оси не видят критериев друг
друга (контракт C3), поэтому и обучаются они на раздельных промптах.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_reliability.methods.m3.axes import (
    AXES,
    axis_anchor,
    build_axis_prompt,
    parse_axis_verdict,
    parse_marker,
)
from rag_reliability.schema import RagSample

#: ``direct`` — только вердикт; ``marker`` — вердикт плюс код маркера кураторов.
MODES: tuple[str, ...] = ("direct", "marker")

_NO_MARKER = "none"
_AXIS_LABEL = {"faithfulness": "faithfulness", "relevance": "relevance"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeExample:
    """Один обучающий пример: промпт оси плюс эталонное завершение."""

    sample_id: str
    axis: str
    system: str
    user: str
    completion: str
    label: int

    @property
    def messages(self) -> list[dict[str, str]]:
        """Чат-формат ровно того же вида, что уходит в клиент на инференсе."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def axis_label(sample: RagSample, axis: str) -> int:
    """Золотая метка нужной оси; иных источников метки у обучения нет."""
    if axis not in _AXIS_LABEL:
        raise ValueError(f"Unknown axis {axis!r}, expected one of {AXES}")
    return int(getattr(sample, _AXIS_LABEL[axis]))


def gold_marker(sample: RagSample, label: int) -> str:
    """Код маркера для завершения в режиме ``marker``.

    Прошедшая ось маркера не несёт: маркер кураторов описывает дефект целиком,
    и приписывать его прошедшей оси значило бы учить модель противоречию.
    """
    if label == 1:
        return _NO_MARKER
    return sample.marker or "unknown"


def completion_text(axis: str, label: int, *, mode: str = "direct", marker: str | None = None) -> str:
    """Эталонное завершение: та же разметка строк, что ждёт парсер C3.

    Строка вердикта — последняя: позиция логпроба ищется после якоря, и хвост
    после вердикта только удлинял бы генерацию на инференсе.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label!r}")
    lines = []
    if mode == "marker":
        lines.append(f"MARKER: {marker or _NO_MARKER}")
    lines.append(f"{axis_anchor(axis)}: {'PASS' if label == 1 else 'FAIL'}")
    return "\n".join(lines)


def check_format_symmetry(example: JudgeExample, *, mode: str = "direct") -> None:
    """Завершение обязано читаться тем же парсером, что и ответ модели.

    Проверяются три вещи, каждая из которых уже ломала прогон: якорь оси есть в
    system (иначе логпробам не за что зацепиться), вердикт в завершении
    разбирается обратно в ту же метку, и в режиме ``marker`` строка маркера
    действительно распознаётся.
    """
    anchor = axis_anchor(example.axis)
    if anchor not in example.system:
        raise ValueError(
            f"Sample {example.sample_id!r} axis {example.axis!r}: the system prompt never asks "
            f"for the {anchor}: line — verdict logprobs would have no anchor at inference"
        )
    parsed = parse_axis_verdict(example.completion, example.axis)
    if parsed is None:
        raise ValueError(
            f"Sample {example.sample_id!r} axis {example.axis!r}: training completion "
            f"{example.completion!r} does not parse back to a verdict"
        )
    if parsed != example.label:
        raise ValueError(
            f"Sample {example.sample_id!r} axis {example.axis!r}: completion parses back to "
            f"{parsed}, but the gold label is {example.label}"
        )
    if mode == "marker" and parse_marker(example.completion) is None:
        raise ValueError(
            f"Sample {example.sample_id!r} axis {example.axis!r}: marker mode completion "
            f"{example.completion!r} has no MARKER: line"
        )


def build_examples(
    samples: Sequence[RagSample],
    *,
    mode: str = "direct",
    prompts_dir: str | Path | None = None,
    axes: Sequence[str] = AXES,
    check_symmetry: bool = True,
) -> list[JudgeExample]:
    """По два примера на кейс — faithfulness и relevance — в формате инференса."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    unknown = [axis for axis in axes if axis not in AXES]
    if unknown:
        raise ValueError(f"Unknown axis/axes {unknown}, expected a subset of {AXES}")

    examples: list[JudgeExample] = []
    for sample in samples:
        for axis in axes:
            label = axis_label(sample, axis)
            system, user = build_axis_prompt(sample, axis, prompts_dir=prompts_dir)
            example = JudgeExample(
                sample_id=sample.id,
                axis=axis,
                system=system,
                user=user,
                completion=completion_text(
                    axis, label, mode=mode, marker=gold_marker(sample, label)
                ),
                label=label,
            )
            if check_symmetry:
                check_format_symmetry(example, mode=mode)
            examples.append(example)
    return examples


def class_balance(examples: Sequence[JudgeExample]) -> dict[str, dict[str, int]]:
    """Сколько PASS и FAIL в каждой оси. Из-за этого числа прогон и схлопывался."""
    balance: dict[str, dict[str, int]] = {}
    for example in examples:
        counts = balance.setdefault(example.axis, {"pass": 0, "fail": 0})
        counts["pass" if example.label == 1 else "fail"] += 1
    return balance


def compute_pos_weight(labels: Sequence[int], mode: str = "balanced") -> float:
    """``n_neg / n_pos`` — вес класса PASS в лоссе; ``none`` — единица.

    Та же формула, что у энкодера (``methods/encoder/train.py``): у методов не
    должно расходиться понятие «взвешенный лосс».
    """
    if mode not in ("none", "balanced"):
        raise ValueError(f"pos_weight_mode must be 'none' or 'balanced', got {mode!r}")
    if mode == "none":
        return 1.0
    positives = sum(labels)
    return (len(labels) - positives) / max(positives, 1)


def oversample_negatives(
    examples: Sequence[JudgeExample], *, seed: int = 42
) -> list[JudgeExample]:
    """Доложить FAIL-примеры каждой оси до паритета с PASS.

    Дисбаланс 72/28 на корпусе кураторов уже приводил к схлопыванию судьи в
    константный вердикт (1,1). Взвешенный лосс правит градиент, oversampling —
    состав батча; в паре они дают ощутимо более устойчивый прогон, чем поодиночке.
    Порядок перемешивается детерминированно: сид входит в ``run.yaml``.
    """
    import random  # noqa: PLC0415

    rng = random.Random(seed)
    out = list(examples)
    for axis in sorted({example.axis for example in examples}):
        axis_examples = [example for example in examples if example.axis == axis]
        negatives = [example for example in axis_examples if example.label == 0]
        positives = [example for example in axis_examples if example.label == 1]
        if not negatives or len(negatives) >= len(positives):
            continue
        deficit = len(positives) - len(negatives)
        out.extend(rng.choices(negatives, k=deficit))
        logger.info(
            "axis %s: oversampled %d FAIL example(s) to match %d PASS",
            axis,
            deficit,
            len(positives),
        )
    rng.shuffle(out)
    return out
