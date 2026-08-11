"""NLI-обоснованность ответа относительно retrieved-чанков.

Заменяет SelfCheck-ветку Метода 6: премиса — чанк контекста, гипотеза —
предложение ответа. Генератор из петли исключён полностью, поэтому вопрос о
валидности прокси-модели (её промпт недоступен, и она измеряла расхождение двух
разных систем, а не галлюцинацию бота) не возникает. Стоимость падает с ~404 000
NLI-пар на корпус до ~90 000: пар на кейс ровно ``n_sentences × n_chunks``,
генераций — ноль.

Основания решения — карточка C4 «Решение» и спека 40_PHASE3 §0.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from rag_reliability.methods.m6.features import razdel_sentences

#: Ровно те 8 ключей, которые метод обязан выдать. Список фиксирован карточкой:
#: молчаливое расширение набора сломало бы контракт артефакта.
GROUNDING_KEYS: tuple[str, ...] = (
    "m6.max_entail",
    "m6.min_entail",
    "m6.mean_entail",
    "m6.mean_contra",
    "m6.max_contra",
    "m6.frac_unsupported",
    "m6.n_sentences",
    "m6.chunk_spread",
)

SentenceSplitter = Callable[[str], list[str]]

DEFAULT_ENTAIL_THRESHOLD = 0.5


class NLILike:
    """Протокол NLI-скорера: ``score(pairs) -> [{'entail': p, 'contra': p}]``."""

    def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class GroundingResult:
    """Фичи вместе с матрицей, на которой они посчитаны.

    Матрица нужна coverage (C4 §3): область покрытия ограничивается чанками,
    которые grounding пометил источниками опоры. Считать coverage по всем 5–8
    чанкам бессмысленно — там много условий, не относящихся к вопросу. Отдавая
    матрицу наружу, coverage не делает ни одного своего NLI-вызова.
    """

    sentences: list[str]
    entail: np.ndarray
    contra: np.ndarray
    features: dict[str, float]
    source_chunk_ids: set[int]


def score_matrix(
    sentences: Sequence[str], chunks: Sequence[str], nli: NLILike
) -> tuple[np.ndarray, np.ndarray]:
    """Матрицы (n_sents, n_chunks) вероятностей entail и contra.

    Все пары кейса уходят в NLI одним вызовом: батчинг окон — забота скорера,
    а покейсовое дробление вызовов удваивало бы накладные расходы.
    """
    if not chunks:
        raise ValueError("grounding needs at least one context chunk, got no context chunks")
    if not sentences:
        raise ValueError("grounding needs at least one answer sentence, got none")
    pairs = [(chunk, sentence) for sentence in sentences for chunk in chunks]
    scored = nli.score(pairs)
    if len(scored) != len(pairs):
        raise ValueError(
            f"NLI returned {len(scored)} row(s) for {len(pairs)} pair(s); "
            "the scorer must preserve pair count and order"
        )
    entail = np.array([row["entail"] for row in scored], dtype=float)
    contra = np.array([row["contra"] for row in scored], dtype=float)
    shape = (len(sentences), len(chunks))
    return entail.reshape(shape), contra.reshape(shape)


def compute_grounding(
    answer: str,
    chunks: Sequence[str],
    nli: NLILike,
    *,
    sentence_splitter: SentenceSplitter = razdel_sentences,
    entail_threshold: float = DEFAULT_ENTAIL_THRESHOLD,
) -> GroundingResult:
    """Фичи опоры ответа на контекст плюс матрица и множество чанков-источников."""
    sentences = sentence_splitter(answer)
    if not sentences:
        raise ValueError("sentence_splitter returned no sentences; a case cannot be dropped")
    entail, contra = score_matrix(sentences, chunks, nli)

    # Агрегация по чанкам: достаточно одного источника опоры, поэтому max по
    # entail; для contra берётся худший чанк — противоречие хотя бы одному
    # источнику это уже противоречие.
    per_sentence_entail = entail.max(axis=1)
    per_sentence_contra = contra.max(axis=1)
    best_chunk = entail.argmax(axis=1)

    features = {
        "m6.max_entail": float(per_sentence_entail.max()),
        # Слабейшее звено: одно неподкреплённое предложение делает ответ неверным,
        # даже если остальные пять подкреплены. Среднее это размывает.
        "m6.min_entail": float(per_sentence_entail.min()),
        "m6.mean_entail": float(per_sentence_entail.mean()),
        "m6.mean_contra": float(per_sentence_contra.mean()),
        "m6.max_contra": float(per_sentence_contra.max()),
        "m6.frac_unsupported": float((per_sentence_entail < entail_threshold).mean()),
        "m6.n_sentences": float(len(sentences)),
        # Сколько разных чанков служат опорой разным предложениям. Высокое
        # значение при коротком ответе — подпись reason_chunk_fact_mixup.
        "m6.chunk_spread": float(len(set(best_chunk.tolist()))),
    }
    return GroundingResult(
        sentences=list(sentences),
        entail=entail,
        contra=contra,
        features=features,
        source_chunk_ids={int(index) for index in best_chunk.tolist()},
    )


def grounding_features(
    answer: str,
    chunks: Sequence[str],
    nli: NLILike,
    *,
    sentence_splitter: SentenceSplitter = razdel_sentences,
    entail_threshold: float = DEFAULT_ENTAIL_THRESHOLD,
) -> dict[str, float]:
    """Опора ответа на контекст: ровно 8 фич из ``GROUNDING_KEYS``.

    E, C: матрицы (n_sents, n_chunks);
    per_sent_entail = E.max(axis=1) — лучший источник для предложения;
    per_sent_contra = C.max(axis=1) — худшее противоречие.
    """
    return compute_grounding(
        answer,
        chunks,
        nli,
        sentence_splitter=sentence_splitter,
        entail_threshold=entail_threshold,
    ).features
