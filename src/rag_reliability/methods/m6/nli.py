"""Оконный NLI-скорер для Метода 6: P(entail) и P(contradict) для пар текстов.

Премиса в grounding-схеме — чанк контекста (медиана 6.4k символов на кейс,
до 8 чанков), гипотеза — предложение ответа. Премиса систематически не влезает
в ``max_length``, поэтому режется на перекрывающиеся окна.

Четыре свойства здесь — починки известных багов, а не оптимизация:

* окна вместо ``truncation=True``: усечённый чанк молча терял условие, ради
  которого его и проверяют;
* бюджет окна считается от **фактической** длины гипотезы; премиса никогда не
  усекается до фиксированного огрызка — не хватило бюджета, значит больше окон;
* пара ``(entail, contra)`` берётся **из одного окна** — того, где максимален
  ``entail``. Два независимых максимума давали сумму больше единицы, и любая
  последующая нормировка теряла смысл;
* вероятности нормируются по паре {entail, contra}, neutral выбрасывается
  (конвенция SelfCheckGPT). При трёхклассовом softmax neutral забирал массу, и
  медиана ``contra`` по корпусу составляла 0.0099 — сигнал был неотличим от нуля.

Чистая логика (нарезка, агрегация, нормировка, ``score_pairs``) не импортирует
torch: она тестируется на мок-токенайзере и мок-forward без весов и GPU.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol

#: Запас под спецтокены пары ([CLS] / [SEP] и т.п.).
SPECIAL_TOKENS_MARGIN = 4
#: Минимум токенов премисы в окне. Если гипотеза настолько длинна, что окна
#: премисы схлопываются ниже этого, усекается ГИПОТЕЗА: сравнивать нечего,
#: когда от источника остаётся огрызок.
MIN_PREMISE_BUDGET = 16

Score = dict[str, float]
#: forward принимает батч пар (premise_window, hypothesis) и возвращает логиты.
ForwardFn = Callable[[list[tuple[str, str]]], Sequence[Sequence[float]]]


class Tokenizer(Protocol):
    """Минимум от HF-токенайзера, нужный оконной нарезке."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def decode(self, token_ids: Any) -> str: ...


def split_tokens(tokens: list[Any], budget: int, overlap: int) -> list[list[Any]]:
    """Режет токены на окна длины <= budget с шагом ``budget - overlap``.

    Последнее окно всегда покрывает хвост: потерянный хвост премисы — это
    потерянное условие, а именно их и ищет grounding.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if overlap < 0 or overlap >= budget:
        raise ValueError(f"overlap must be in [0, budget), got {overlap} with budget {budget}")
    if len(tokens) <= budget:
        return [tokens]
    stride = budget - overlap
    windows: list[list[Any]] = []
    for start in range(0, len(tokens), stride):
        windows.append(tokens[start : start + budget])
        if start + budget >= len(tokens):
            break
    return windows


def _token_ids(text: str, tokenizer: Tokenizer) -> list[Any]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def fit_hypothesis(hypothesis: str, *, tokenizer: Tokenizer, max_length: int) -> tuple[str, int]:
    """Гипотеза, гарантированно оставляющая премисе ``MIN_PREMISE_BUDGET`` токенов.

    Возвращает (текст гипотезы, длина в токенах). Усечение здесь — единственное
    во всём модуле, и оно касается гипотезы, а не источника.
    """
    ids = _token_ids(hypothesis, tokenizer)
    cap = max_length - SPECIAL_TOKENS_MARGIN - MIN_PREMISE_BUDGET
    if cap < 1:
        raise ValueError(
            f"max_length={max_length} is too small for a hypothesis and "
            f"{MIN_PREMISE_BUDGET} premise tokens"
        )
    if len(ids) <= cap:
        return hypothesis, len(ids)
    return tokenizer.decode(ids[:cap]), cap


def window_premise(
    premise: str,
    hypothesis: str,
    *,
    tokenizer: Tokenizer,
    max_length: int,
    overlap: int,
) -> list[str]:
    """Окна премисы, каждое из которых влезает в пару с гипотезой.

    Бюджет считается от фактической длины гипотезы; премиса не усекается
    никогда — вместо этого растёт число окон.
    """
    _, n_hypothesis = fit_hypothesis(hypothesis, tokenizer=tokenizer, max_length=max_length)
    budget = max_length - n_hypothesis - SPECIAL_TOKENS_MARGIN
    premise_ids = _token_ids(premise, tokenizer)
    if not premise_ids:
        return [premise]
    if len(premise_ids) <= budget:
        return [premise]
    effective_overlap = min(overlap, budget - 1)
    return [
        tokenizer.decode(window)
        for window in split_tokens(premise_ids, budget, effective_overlap)
    ]


def two_class_probabilities(
    logits: Sequence[float], *, entail_index: int, contra_index: int
) -> Score:
    """Softmax по паре {entail, contra}: neutral выбрасывается.

    Конвенция SelfCheckGPT-NLI. Трёхклассовая нормировка отдаёт neutral
    основную массу на длинных банковских премисах, и contra перестаёт быть
    сигналом вообще.
    """
    entail_logit = float(logits[entail_index])
    contra_logit = float(logits[contra_index])
    shift = max(entail_logit, contra_logit)
    entail_exp = math.exp(entail_logit - shift)
    contra_exp = math.exp(contra_logit - shift)
    total = entail_exp + contra_exp
    entail = entail_exp / total
    return {"entail": entail, "contra": 1.0 - entail}


def aggregate_windows(scores: list[Score], groups: list[int], n_pairs: int) -> list[Score]:
    """Схлопывает окна в скор пары: побеждает окно с максимальным ``entail``.

    ``contra`` берётся из ТОГО ЖЕ окна. Независимые максимумы по двум классам
    описывают окно, которого не существует, и ломают нормировку: сумма пары
    переставала быть единицей.
    """
    best: list[Score | None] = [None] * n_pairs
    for score, group in zip(scores, groups, strict=True):
        current = best[group]
        if current is None or score["entail"] > current["entail"]:
            best[group] = dict(score)
    missing = [index for index, score in enumerate(best) if score is None]
    if missing:
        raise ValueError(
            f"no window scores for {len(missing)} pair(s): {missing[:5]}; "
            "every pair must produce at least one window"
        )
    return [score for score in best if score is not None]


def score_pairs(
    pairs: list[tuple[str, str]],
    *,
    tokenizer: Tokenizer,
    forward_fn: ForwardFn,
    entail_index: int,
    contra_index: int,
    max_length: int,
    overlap: int,
    batch_size: int,
) -> list[Score]:
    """Скоры пар (premise, hypothesis): ровно один dict на пару, порядок сохранён.

    Батчуются ОКНА, а не пары: у длинного чанка их десятки, и батч по парам
    оставлял бы модель почти без работы.
    """
    expanded: list[tuple[str, str]] = []
    groups: list[int] = []
    for index, (premise, hypothesis) in enumerate(pairs):
        fitted, _ = fit_hypothesis(hypothesis, tokenizer=tokenizer, max_length=max_length)
        for window in window_premise(
            premise, hypothesis, tokenizer=tokenizer, max_length=max_length, overlap=overlap
        ):
            expanded.append((window, fitted))
            groups.append(index)

    scores: list[Score] = []
    for start in range(0, len(expanded), batch_size):
        batch = expanded[start : start + batch_size]
        for row in forward_fn(batch):
            scores.append(
                two_class_probabilities(row, entail_index=entail_index, contra_index=contra_index)
            )
    if not pairs:
        return []
    return aggregate_windows(scores, groups, len(pairs))


class NLIScorer:
    """Обёртка над мультиязычной sequence-classification NLI-моделью.

    Счётчик ``n_pairs`` / ``n_windows`` — не отладка: стоимость ветки (число
    NLI-пар на корпус) входит в её обоснование, и считать её постфактум по
    логам нечем.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 512,
        overlap: int = 128,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        id2label = {int(index): label.lower() for index, label in self.model.config.id2label.items()}
        self.entail_index = next(index for index, label in id2label.items() if "entail" in label)
        self.contra_index = next(index for index, label in id2label.items() if "contra" in label)
        self.batch_size = batch_size
        self.max_length = max_length
        self.overlap = overlap
        self.n_pairs = 0
        self.n_windows = 0

    def _forward(self, batch: list[tuple[str, str]]) -> list[list[float]]:
        encoded = self.tokenizer(
            [premise for premise, _ in batch],
            [hypothesis for _, hypothesis in batch],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**encoded).logits.float()
        self.n_windows += len(batch)
        return logits.tolist()

    def score(self, pairs: list[tuple[str, str]]) -> list[Score]:
        self.n_pairs += len(pairs)
        return score_pairs(
            pairs,
            tokenizer=self.tokenizer,
            forward_fn=self._forward,
            entail_index=self.entail_index,
            contra_index=self.contra_index,
            max_length=self.max_length,
            overlap=self.overlap,
            batch_size=self.batch_size,
        )
