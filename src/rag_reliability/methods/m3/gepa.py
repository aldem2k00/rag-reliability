"""GEPA prompt evolution for the Method 3 judge — переработка по карточке D2.

Прошлая постановка была не «проверена и отвергнута», а недофинансирована и
закрыта по неинформативному сигналу. Что изменено здесь:

* **метрика** — balanced accuracy оси вместо accuracy по двум осям. При базовой
  ставке reliable 72% точность почти плоская: кандидат улучшает её, чаще отвечая
  PASS, и одновременно ухудшает отчётный macro-F1. Balanced accuracy не зависит
  от калибровки (в отличие от macro-F1 с фиксированным порогом), которая внутри
  эволюции дрейфует;
* **данные** — D_pareto из ``folds.json``: 300 кейсов со стратификацией по классу,
  только из train-части текущего фолда. Ни один held-out кейс в оптимизацию не
  попадает, и это проверяется в рантайме, а не только в тестах;
* **feedback** — тип ошибки, фрагмент ответа и ближайший чанк. Рефлексирующая LM
  раньше узнавала *что*, но не *почему*;
* **синхронизация** — GEPA эволюционирует ту же system-инструкцию оси, которую
  читает инференс (``configs/prompts/{axis}.yaml`` → ``--prompt-file-{axis}``),
  и разбирает вывод тем же ``parse_axis_verdict``. Раньше эволюционировалась
  отдельная DSPy-сигнатура с другим форматом вывода, и regex-фолбэк инференса
  ломался;
* **санитария** — детектор посторонних доменов: в закоммиченных эволюционировавших
  инструкциях жили телескоп «Хаббл», соул-джаз и Парфенон, то есть содержимое
  псевдо-корпуса;
* **стоп-правило H5** — отвергаем только при верхней границе 95% ДИ приращения
  ниже нуля. Прежний отказ был сделан по точечной оценке при ДИ, накрывающем ноль.

Чистые функции (метрика, feedback, отбор данных, санитария, стоп-правило) не
требуют dspy и покрыты тестами; dspy импортируется лениво внутри функций,
которым он нужен (extra ``gepa``).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rag_reliability.evaluation.bootstrap import PairedResult
from rag_reliability.methods.m3.axes import (
    AXES,
    AXIS_FAITHFULNESS,
    AXIS_RELEVANCE,
    axis_anchor,
    load_axis_prompt,
    parse_axis_verdict,
)
from rag_reliability.schema import RagSample
from rag_reliability.splits import extract_last_client_turn
from rag_reliability.thresholds import macro_f1_binary

#: Значения gold-маркера, означающие «маркера у кейса нет».
_NO_MARKER = (None, "", "none", "unknown")

#: Размер Pareto-набора GEPA (карточка D2 §2). При ``val_size = 30`` гранулярность
#: скора равна 1/60, и «лучший» кандидат отличался от сида на два решения.
DEFAULT_PARETO_SIZE = 300
DEFAULT_TRAIN_SIZE = 300

#: Сколько символов ответа цитировать в feedback.
DEFAULT_SNIPPET_CHARS = 240

#: Токены короче этого в лексическом сопоставлении с чанками игнорируются:
#: предлоги и союзы иначе перевешивают термины.
_MIN_TOKEN_LEN = 4

_WORD_RE = re.compile(r"\w+", re.UNICODE)

#: Разметка чанков: корпус организаторов пишет ``[CHUNK n]``, эволюционировавшие
#: инструкции — ``[Чанк n]``. Поддержаны обе.
_CHUNK_MARKER = re.compile(r"\[(?:CHUNK|ЧАНК|ФРАГМЕНТ)\s*(\d+)\]", re.IGNORECASE)

#: Демонстрация пустого чанка в промпте: судью учат, что чанки бывают пустыми.
_EMPTY_CHUNK_DEMO = re.compile(r"\[(?:CHUNK|ЧАНК|ФРАГМЕНТ)\s*\d+\]\s*\(\s*\.\.\.\s*\)", re.IGNORECASE)

#: Стоп-слова псевдо-корпуса: сюжеты, которых в банковском корпусе быть не может.
#: Список — константа модуля, а не конфиг, потому что D2 не владеет ``configs/``;
#: переопределяется файлом через ``--stopwords-file`` (см. ``load_stopwords``).
#: Основа — то, что реально вшилось в закоммиченные эволюционировавшие инструкции.
#: Термины подобраны так, чтобы не быть подстрокой банковского слова: «афин»
#: сюда не годится, потому что срабатывает на «парафин».
FOREIGN_DOMAIN_TERMS: tuple[str, ...] = (
    "хаббл",
    "телескоп",
    "соул-джаз",
    "соул джаз",
    "джаз",
    "парфенон",
    "википеди",
    "фотосинтез",
    "динозавр",
    "галактик",
    "эверест",
    "шекспир",
    "моцарт",
)


# --------------------------------------------------------------------------- #
# Метрика: balanced accuracy
# --------------------------------------------------------------------------- #


def axis_gold(sample: RagSample, axis: str) -> int:
    """Gold-метка нужной оси. Оси независимы — ``reliable`` здесь не при чём."""
    if axis == AXIS_FAITHFULNESS:
        return int(sample.faithfulness)
    if axis == AXIS_RELEVANCE:
        return int(sample.relevance)
    raise ValueError(f"Unknown axis {axis!r}, expected one of {AXES}")


def verdict(value: int) -> str:
    return "PASS" if value == 1 else "FAIL"


def has_marker(sample: RagSample) -> bool:
    return sample.marker not in _NO_MARKER


def balanced_accuracy(golds: Sequence[int], preds: Sequence[int | None]) -> float:
    """``0.5 * (recall_pos + recall_neg)``.

    Почему не accuracy: при 72% позитивов константный PASS даёт 0.72 и выглядит
    сильным кандидатом. Здесь он даёт ровно 0.5 при любом дисбалансе, поэтому
    эволюция не может выиграть, сдвинув судью в режим «всегда PASS».

    ``None`` в ``preds`` — вердикта в выводе не нашлось; это ошибка, а не
    пропуск: сломанный формат вывода обязан стоить кандидату скора, иначе GEPA
    свободно эволюционирует промпт, который инференс не сможет разобрать.
    """
    if len(golds) != len(preds):
        raise ValueError(
            f"golds and preds must have equal lengths, got {len(golds)} and {len(preds)}"
        )
    if not golds:
        raise ValueError("Cannot compute balanced accuracy on an empty set")
    recalls: list[float] = []
    for label in (0, 1):
        indices = [index for index, gold in enumerate(golds) if gold == label]
        if not indices:
            raise ValueError(
                f"Class {label} is absent from the {len(golds)} gold label(s); balanced "
                "accuracy is undefined — stratify the optimization set"
            )
        hits = sum(1 for index in indices if preds[index] == label)
        recalls.append(hits / len(indices))
    return 0.5 * (recalls[0] + recalls[1])


#: Публичное имя метрики GEPA из карточки D2 §1.
gepa_metric = balanced_accuracy


def class_weights(golds: Sequence[int], *, normalize: bool = False) -> dict[int, float]:
    """Вес класса ``n / (2 * n_class)``.

    Существует ради DSPy: GEPA агрегирует метрику как СРЕДНЕЕ по-примерных
    скоров, а balanced accuracy — величина уровня набора. С этими весами
    ``mean(example_score) == balanced_accuracy`` тождественно (проверяется
    тестом), то есть по-примерный скор GEPA и отчётная метрика — одно и то же
    число, а не две похожие величины.

    ``normalize`` делит веса на максимальный, загоняя по-примерный скор в
    [0, 1]. Это нужно на реальном дисбалансе: при 72/28 вес редкого класса равен
    1.79, и если GEPA где-то клипует скор в [0, 1], награда за редкий класс
    срезается — то есть ровно то свойство, ради которого метрику и меняли.
    Деление на константу монотонно, поэтому порядок кандидатов не меняется, а
    среднее остаётся пропорциональным balanced accuracy.
    """
    if not golds:
        raise ValueError("Cannot compute class weights on an empty set")
    total = len(golds)
    weights: dict[int, float] = {}
    for label in (0, 1):
        count = sum(1 for gold in golds if gold == label)
        if count == 0:
            raise ValueError(
                f"Class {label} is absent from the {total} gold label(s); the optimization "
                "set must contain both classes"
            )
        weights[label] = total / (2.0 * count)
    if normalize:
        peak = max(weights.values())
        weights = {label: value / peak for label, value in weights.items()}
    return weights


def example_score(gold: int, pred: int | None, weights: Mapping[int, float]) -> float:
    """Скор одного примера для GEPA: попадание, взвешенное обратной частотой класса."""
    if gold not in weights:
        raise ValueError(f"No class weight for gold label {gold!r} (known: {sorted(weights)})")
    return weights[gold] if pred == gold else 0.0


# --------------------------------------------------------------------------- #
# Feedback с диагностикой
# --------------------------------------------------------------------------- #


def load_marker_gloss(path: str | Path) -> dict[str, str]:
    """Глоссы маркеров из ``configs/markers.yaml``.

    Не хардкод: на реальном корпусе файл заменяется словарём кураторов без
    правки кода.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(key): str(value) for key, value in data.items()}


def answer_snippet(answer: str, max_chars: int = DEFAULT_SNIPPET_CHARS) -> str:
    """Начало ответа одной строкой — то, о чём идёт речь в feedback."""
    collapsed = " ".join(answer.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[:max_chars].rstrip() + "…"


def split_context_chunks(context: str) -> list[tuple[int, str]]:
    """``[(номер чанка, текст)]`` по маркерам ``[CHUNK n]`` / ``[Чанк n]``.

    Контекст в корпусе — склеенная строка, отдельного поля чанков нет.
    """
    markers = list(_CHUNK_MARKER.finditer(context))
    if not markers:
        return []
    chunks: list[tuple[int, str]] = []
    for index, match in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(context)
        chunks.append((int(match.group(1)), context[match.end() : end].strip()))
    return chunks


def _content_tokens(text: str) -> set[str]:
    return {
        token.lower() for token in _WORD_RE.findall(text) if len(token) >= _MIN_TOKEN_LEN
    }


def closest_chunk(answer: str, context: str) -> int | None:
    """Номер чанка с наибольшим лексическим пересечением с ответом.

    Дешёвая эвристика вместо NLI: feedback читает рефлексирующая LM, которой
    нужен указатель «смотри сюда», а не вердикт. ``None`` — чанков нет или ни
    один не пересекается с ответом (это тоже диагноз).
    """
    chunks = split_context_chunks(context)
    if not chunks:
        return None
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return None
    best_number, best_overlap = None, 0
    for number, text in chunks:
        overlap = len(answer_tokens & _content_tokens(text))
        if overlap > best_overlap:
            best_number, best_overlap = number, overlap
    return best_number


def gepa_feedback(
    *,
    axis: str,
    gold: int,
    pred: int | None,
    answer: str,
    context: str = "",
    question: str = "",
    marker: str | None = None,
    use_markers: bool = False,
    gloss: Mapping[str, str] | None = None,
    max_chars: int = DEFAULT_SNIPPET_CHARS,
) -> str:
    """Текстовый разбор ошибки для рефлексирующей LM.

    Раньше сообщалась только правильная метка: LM узнавала *что*, но не *почему*.
    Добавлены тип ошибки (глосс маркера), фрагмент ответа и указание на
    ближайший чанк — ровно то, чего не хватало.

    Feedback на русском: его читает LM, эволюционирующая русский промпт.
    """
    anchor = axis_anchor(axis)
    if pred == gold:
        return "Верно."

    parts = [f"Ошибка. Правильный ответ: {anchor}={verdict(gold)}."]
    if pred is None:
        parts.append(
            f"Вердикта в выводе не нашлось: строки «{anchor}: PASS» или «{anchor}: FAIL» "
            "в ответе нет. Формат вывода обязателен."
        )
    if use_markers and marker not in _NO_MARKER:
        glossary = gloss or {}
        described = glossary[marker] if marker in glossary else marker
        parts.append(f"Тип ошибки: {described}.")

    parts.append(f"Фрагмент ответа: «{answer_snippet(answer, max_chars)}».")

    if context.strip():
        number = closest_chunk(answer, context)
        where = f"ближайший по лексике — чанк {number}" if number is not None else (
            "ни один чанк лексически не пересекается с этим фрагментом"
        )
        if gold == 0:
            parts.append(f"В чанках это не подтверждается: {where}.")
        else:
            parts.append(f"Этот фрагмент опирается на [CTX]: {where}.")
    elif axis == AXIS_RELEVANCE and question.strip():
        turn = extract_last_client_turn(question) or question
        parts.append(f"Вопрос клиента: «{answer_snippet(turn, max_chars)}».")

    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Отбор данных: D_pareto без утечки из held-out
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OptimizationSets:
    """Наборы GEPA одного фолда.

    ``held_out`` хранится не для использования, а для проверки: инвариант
    «ни один held-out кейс не попал в оптимизацию» проверяется в рантайме
    каждого прогона, а не только в тестах.
    """

    train: list[RagSample]
    pareto: list[RagSample]
    held_out: list[RagSample]
    repeat: int
    fold: int


class LeakageError(ValueError):
    """Held-out кейс попал в оптимизационный набор."""


def fold_partition(
    samples: Sequence[RagSample],
    folds: Mapping[str, Any],
    *,
    repeat: int,
    fold: int,
) -> tuple[list[RagSample], list[RagSample]]:
    """``(train-часть, held-out фолд)`` по ``data/splits/folds.json``.

    Кейсы, отсутствующие в ``assignment`` (oversized-группа, 759 кейсов на
    корпусе организаторов), в held-out не попадают никогда и потому безопасны
    для оптимизации — они уходят в train-часть, как и задумано ``splits.py``.
    """
    assignment = folds["assignment"]
    config = folds["config"]
    n_repeats, n_folds = int(config["n_repeats"]), int(config["n_folds"])
    if not 0 <= repeat < n_repeats:
        raise ValueError(f"repeat must be in [0, {n_repeats}), got {repeat}")
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")

    train: list[RagSample] = []
    held_out: list[RagSample] = []
    for sample in samples:
        if sample.id not in assignment:
            train.append(sample)
            continue
        assigned = assignment[sample.id]
        if len(assigned) != n_repeats:
            raise ValueError(
                f"folds.assignment[{sample.id!r}] has {len(assigned)} repeat(s), "
                f"expected {n_repeats}"
            )
        (held_out if assigned[repeat] == fold else train).append(sample)
    if not held_out:
        raise ValueError(
            f"repeat {repeat} fold {fold} is empty for the {len(samples)} given sample(s); "
            "the folds were built on a different corpus"
        )
    return train, held_out


def stratified_subsample(
    samples: Sequence[RagSample],
    size: int,
    *,
    seed: int,
    label_fn: Callable[[RagSample], int] = lambda sample: sample.reliable,
) -> list[RagSample]:
    """Детерминированная подвыборка со стратификацией по бинарной метке.

    Пропорциональное размещение: доля позитивов подвыборки повторяет долю в
    источнике. Порядок входа не важен — кейсы сортируются по id, поэтому один
    seed даёт один и тот же набор независимо от порядка чтения файла.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    ordered = sorted(samples, key=lambda sample: sample.id)
    if size >= len(ordered):
        return ordered

    by_label: dict[int, list[RagSample]] = {0: [], 1: []}
    for sample in ordered:
        label = int(label_fn(sample))
        if label not in by_label:
            raise ValueError(f"label_fn must return 0 or 1, got {label!r} for {sample.id!r}")
        by_label[label].append(sample)

    rng = random.Random(seed)
    picked: list[RagSample] = []
    # Позитивы округляются вниз, остаток добирается негативами: так итоговый
    # размер точно равен size, а не size ± 1 от двойного округления.
    n_positive = min(len(by_label[1]), int(size * len(by_label[1]) / len(ordered)))
    n_negative = min(len(by_label[0]), size - n_positive)
    n_positive = min(len(by_label[1]), size - n_negative)
    picked += rng.sample(by_label[1], n_positive)
    picked += rng.sample(by_label[0], n_negative)
    picked.sort(key=lambda sample: sample.id)
    rng.shuffle(picked)
    return picked


def build_optimization_sets(
    samples: Sequence[RagSample],
    folds: Mapping[str, Any],
    *,
    axis: str,
    repeat: int = 0,
    fold: int = 0,
    pareto_size: int = DEFAULT_PARETO_SIZE,
    train_size: int = DEFAULT_TRAIN_SIZE,
    seed: int = 0,
) -> OptimizationSets:
    """Train и D_pareto из train-части фолда, стратифицированные по метке оси.

    Наборы не пересекаются между собой и не пересекаются с held-out. Инвариант
    проверяется здесь же и падает :class:`LeakageError` — молча «почти
    правильный» набор хуже упавшего прогона.
    """
    train_part, held_out = fold_partition(samples, folds, repeat=repeat, fold=fold)

    def label(sample: RagSample) -> int:
        return axis_gold(sample, axis)

    pareto = stratified_subsample(train_part, pareto_size, seed=seed, label_fn=label)
    pareto_ids = {sample.id for sample in pareto}
    remainder = [sample for sample in train_part if sample.id not in pareto_ids]
    if not remainder:
        raise ValueError(
            f"D_pareto of {pareto_size} consumed the whole train part "
            f"({len(train_part)} case(s)); nothing left for the reflection minibatches"
        )
    # seed + 1: та же подвыборка с тем же seed из непересекающихся множеств дала
    # бы скоррелированный отбор, а train и pareto должны быть независимы.
    train = stratified_subsample(remainder, train_size, seed=seed + 1, label_fn=label)

    sets = OptimizationSets(
        train=train, pareto=pareto, held_out=held_out, repeat=repeat, fold=fold
    )
    check_no_leakage(sets)
    return sets


def check_no_leakage(sets: OptimizationSets) -> None:
    """Ни один held-out кейс не в train и не в pareto; train и pareto не пересекаются."""
    held_out_ids = {sample.id for sample in sets.held_out}
    train_ids = {sample.id for sample in sets.train}
    pareto_ids = {sample.id for sample in sets.pareto}
    for name, ids in (("train", train_ids), ("pareto", pareto_ids)):
        leaked = sorted(ids & held_out_ids)
        if leaked:
            raise LeakageError(
                f"repeat {sets.repeat} fold {sets.fold}: {len(leaked)} held-out case(s) "
                f"leaked into {name}: {leaked[:5]}"
            )
    overlap = sorted(train_ids & pareto_ids)
    if overlap:
        raise LeakageError(
            f"repeat {sets.repeat} fold {sets.fold}: {len(overlap)} case(s) are in both "
            f"train and D_pareto: {overlap[:5]}"
        )


# --------------------------------------------------------------------------- #
# Санитария промптов
# --------------------------------------------------------------------------- #


def load_stopwords(path: str | Path | None) -> tuple[str, ...]:
    """Стоп-слова посторонних доменов: список из файла либо константа модуля."""
    if path is None:
        return FOREIGN_DOMAIN_TERMS
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("terms")
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            f"Stopwords file {path} must be a non-empty list of terms "
            "(or a mapping with a 'terms' list)"
        )
    return tuple(str(term).strip().lower() for term in payload)


def foreign_domain_hits(text: str, terms: Sequence[str] = FOREIGN_DOMAIN_TERMS) -> list[str]:
    """Сработавшие стоп-слова посторонних доменов, в порядке списка.

    Подстрочное сравнение по нижнему регистру: «Хаббл» надо ловить и в форме
    «Хаббла», а морфологию русского здесь разбирать нечем и незачем.
    """
    lowered = text.lower()
    return [term for term in terms if term in lowered]


def prompt_defects(
    text: str, axis: str, *, terms: Sequence[str] = FOREIGN_DOMAIN_TERMS
) -> list[str]:
    """Претензии к эволюционировавшей инструкции перед сохранением.

    Не исключение: GEPA возвращает то, что нашла, и решение «не публиковать»
    принимает человек. Но молча сохранить промпт с примером про телескоп
    «Хаббл» — как раз то, что уже произошло.
    """
    defects: list[str] = []
    anchor = axis_anchor(axis)
    if anchor not in text:
        defects.append(
            f"инструкция не требует строки «{anchor}: PASS|FAIL» — у логпробов не будет якоря, "
            "а regex-фолбэк не найдёт вердикт"
        )
    hits = foreign_domain_hits(text, terms)
    if hits:
        defects.append(
            "посторонние домены (следы псевдо-корпуса): " + ", ".join(hits)
        )
    if _EMPTY_CHUNK_DEMO.search(text):
        defects.append(
            "демонстрация пустого чанка вида «[Чанк N] (...)»: судью учат, что чанки бывают пустыми"
        )
    return defects


# --------------------------------------------------------------------------- #
# Стоп-правило H5
# --------------------------------------------------------------------------- #

H5_REJECTED = "rejected"
H5_SUPPORTED = "supported"
H5_UNTESTED = "untested"


@dataclass(frozen=True)
class H5Verdict:
    """Вердикт по H5 с явным различением «отвергнута» и «не проверена»."""

    delta: float
    ci95: tuple[float, float]
    p: float
    n_seeds: int
    n_cases: int
    status: str
    conclusion: str

    @property
    def rejected(self) -> bool:
        return self.status == H5_REJECTED


def h5_paired_bootstrap(
    y: Any,
    markers_by_seed: Sequence[Any],
    plain_by_seed: Sequence[Any],
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = macro_f1_binary,
    B: int = 10_000,
    seed: int = 0,
) -> PairedResult:
    """Парный бутстрэп приращения markers − plain по кейсам И по сидам.

    Внутри реплики пересэмплируются кейсы (пары сохраняются: оба варианта видят
    один и тот же набор индексов), а по сидам метрика усредняется. Так в
    интервал попадают обе причины разброса: жребий кейсов и жребий сида.
    """
    if len(markers_by_seed) != len(plain_by_seed):
        raise ValueError(
            f"markers and plain must be paired by seed, got {len(markers_by_seed)} "
            f"and {len(plain_by_seed)} run(s)"
        )
    if not markers_by_seed:
        raise ValueError("H5 needs at least one paired (markers, plain) run")

    y_array = np.asarray(y, dtype=int)
    markers = [np.asarray(values, dtype=int) for values in markers_by_seed]
    plain = [np.asarray(values, dtype=int) for values in plain_by_seed]
    for name, arrays in (("markers", markers), ("plain", plain)):
        bad = [index for index, array in enumerate(arrays) if array.shape != y_array.shape]
        if bad:
            raise ValueError(
                f"{name} run(s) {bad[:5]} disagree with y on the number of cases "
                f"({y_array.shape[0]})"
            )

    def mean_metric(indices: np.ndarray, runs: Sequence[np.ndarray]) -> float:
        sampled_y = y_array[indices]
        return float(np.mean([metric_fn(sampled_y, run[indices]) for run in runs]))

    rng = np.random.default_rng(seed)
    deltas = np.empty(B, dtype=float)
    for replicate in range(B):
        indices = rng.integers(0, len(y_array), size=len(y_array))
        deltas[replicate] = mean_metric(indices, markers) - mean_metric(indices, plain)

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    all_indices = np.arange(len(y_array))
    delta = mean_metric(all_indices, markers) - mean_metric(all_indices, plain)
    if np.all(deltas == 0.0):
        p_value = 1.0
    else:
        p_value = min(
            1.0, 2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))
        )
    return PairedResult(
        delta=delta, ci95=(float(lo), float(hi)), p=p_value, significant=p_value < 0.05
    )


def h5_decision(result: PairedResult, *, n_seeds: int, n_cases: int) -> H5Verdict:
    """Стоп-правило H5 (карточка D2 §8).

    Отвергаем ТОЛЬКО если верхняя граница 95% ДИ приращения (markers − plain)
    ниже нуля. Прежний отказ был сделан по точечной оценке +0.0216 при ДИ
    [−0.037, +0.100]: такой интервал означает «не измерено», а не «не работает».
    """
    lo, hi = result.ci95
    if hi < 0.0:
        status = H5_REJECTED
        conclusion = (
            f"H5 отвергнута: верхняя граница 95% ДИ приращения {hi:+.4f} ниже нуля — "
            "маркерный feedback измеримо вредит."
        )
    elif lo > 0.0:
        status = H5_SUPPORTED
        conclusion = (
            f"H5 подтверждена: нижняя граница 95% ДИ приращения {lo:+.4f} выше нуля."
        )
    else:
        status = H5_UNTESTED
        conclusion = (
            f"H5 не проверена: 95% ДИ приращения [{lo:+.4f}, {hi:+.4f}] накрывает ноль. "
            "Отвергать по точечной оценке нельзя — именно так был получен прошлый "
            "ложный отказ."
        )
    return H5Verdict(
        delta=result.delta,
        ci95=(lo, hi),
        p=result.p,
        n_seeds=n_seeds,
        n_cases=n_cases,
        status=status,
        conclusion=conclusion,
    )


def h5_verdict(
    y: Any,
    markers_by_seed: Sequence[Any],
    plain_by_seed: Sequence[Any],
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = macro_f1_binary,
    B: int = 10_000,
    seed: int = 0,
) -> H5Verdict:
    """Парный бутстрэп + стоп-правило одним вызовом."""
    result = h5_paired_bootstrap(
        y, markers_by_seed, plain_by_seed, metric_fn=metric_fn, B=B, seed=seed
    )
    return h5_decision(result, n_seeds=len(markers_by_seed), n_cases=len(np.asarray(y)))


# --------------------------------------------------------------------------- #
# Связка с DSPy (ленивые импорты)
# --------------------------------------------------------------------------- #


def load_seed_instruction(
    axis: str,
    *,
    prompts_dir: str | Path | None = None,
    markers_path: str | Path | None = None,
) -> str:
    """Сид эволюции — system-инструкция оси из ``configs/prompts/{axis}.yaml``.

    Тот же текст, который читает инференс: чеклист маркеров уже подставлен
    загрузчиком C3. Раньше сидом была отдельная DSPy-сигнатура, и эволюция
    улучшала промпт, которого инференс никогда не видел.
    """
    return load_axis_prompt(axis, prompts_dir=prompts_dir, markers_path=markers_path).system


def build_program(axis: str, instruction: str):
    """DSPy-модуль с ОДНИМ строковым выходом — сырым ответом судьи.

    Вывод разбирается тем же ``parse_axis_verdict``, что и в инференсе, а
    инструкция — тот же system-промпт оси. Типизированные поля
    ``faithfulness/relevance`` из прошлой версии как раз и разводили обучение с
    инференсом: DSPy печатал свои адаптерные префиксы, инференс ждал строку
    ``FAITHFULNESS: PASS``, и regex-фолбэк ломался.

    Остаточное расхождение, которое в рамках DSPy не убрать: chat-адаптер
    оборачивает поля своими маркерами. Формат самих вердиктных строк совпадает.
    """
    import dspy  # noqa: PLC0415

    anchor = axis_anchor(axis)

    class AxisJudge(dspy.Signature):
        """(инструкция подставляется через with_instructions)"""

        request: str = dspy.InputField(desc="запрос судье в формате оси ([Q]/[CTX]/[A])")
        judgement: str = dspy.OutputField(
            desc=f"ANALYSIS / MARKER / {anchor}: PASS или FAIL — ровно три строки"
        )

    return dspy.Predict(AxisJudge.with_instructions(instruction))


def build_examples(
    samples: Sequence[RagSample],
    axis: str,
    *,
    prompts_dir: str | Path | None = None,
    max_context_chars: int | None = None,
) -> list:
    """``dspy.Example`` с тем же user-промптом, который строит инференс."""
    import dspy  # noqa: PLC0415

    from rag_reliability.methods.m3.axes import build_axis_prompt  # noqa: PLC0415

    spec = load_axis_prompt(axis, prompts_dir=prompts_dir)
    examples = []
    for sample in samples:
        _, request = build_axis_prompt(
            sample, axis, spec=spec, max_context_chars=max_context_chars
        )
        examples.append(
            dspy.Example(
                request=request,
                gold_label=axis_gold(sample, axis),
                marker=sample.marker,
                answer=sample.answer,
                context=sample.context if spec.needs_context else "",
                question=sample.question,
            ).with_inputs("request")
        )
    return examples


def make_metric(
    axis: str,
    *,
    use_markers: bool,
    gloss: Mapping[str, str],
    weights: Mapping[int, float],
    max_chars: int = DEFAULT_SNIPPET_CHARS,
):
    """Метрика в сигнатуре ``dspy.GEPA``: взвешенный скор + диагностический feedback."""
    import dspy  # noqa: PLC0415

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):  # noqa: ARG001
        gold_label = int(gold.gold_label)
        text = str(getattr(pred, "judgement", "") or "")
        predicted = parse_axis_verdict(text, axis)
        score = example_score(gold_label, predicted, weights)
        feedback = gepa_feedback(
            axis=axis,
            gold=gold_label,
            pred=predicted,
            answer=str(getattr(gold, "answer", "")),
            context=str(getattr(gold, "context", "") or ""),
            question=str(getattr(gold, "question", "") or ""),
            marker=getattr(gold, "marker", None),
            use_markers=use_markers,
            gloss=gloss,
            max_chars=max_chars,
        )
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


def evaluate_program(program, examples: Sequence[Any], axis: str) -> float:
    """Balanced accuracy модуля на наборе — тот же скор, что оптимизирует GEPA."""
    golds = [int(example.gold_label) for example in examples]
    preds = [
        parse_axis_verdict(str(getattr(program(request=example.request), "judgement", "")), axis)
        for example in examples
    ]
    return balanced_accuracy(golds, preds)


def extract_instruction(program) -> str:
    """Инструкция единственного предиктора оптимизированного модуля."""
    _, predictor = next(iter(program.named_predictors()))
    return predictor.signature.instructions


def serialize_detailed(dr: Any) -> dict:
    """``track_stats`` → json-совместимый дайджест (кандидаты, скоры, счётчики)."""
    if dr is None:
        return {}
    try:
        return json.loads(json.dumps(dr.to_dict(), default=str, ensure_ascii=False))
    except Exception:
        return {
            "val_aggregate_scores": getattr(dr, "val_aggregate_scores", None),
            "best_idx": getattr(dr, "best_idx", None),
            "total_metric_calls": getattr(dr, "total_metric_calls", None),
            "candidates": [
                {k: str(v) for k, v in c.items()} if isinstance(c, dict) else str(c)
                for c in (getattr(dr, "candidates", None) or [])
            ],
        }
