"""Self-consistency Метода 3: N сэмплов одной оси вместо одного T=0 вызова.

Зачем. У 50-79% кейсов ``p_faith > 0.99``: масса связок физически ограничивает
AUC, ранжировать нечего. N сэмплов при T>0 дают настоящую гранулярность ранга на
том же промпте и той же модели.

Инфраструктура уже есть: ``LLMClient.chat(n=...)`` умеет n>1 и деградирует в
последовательные вызовы. Этот модуль — обёртка поверх публичного ``chat``;
``judge_client.py`` (владение A4) не меняется.

Усредняются ВЕРОЯТНОСТИ, а не голоса: голоса дают гранулярность 1/N, вероятности
непрерывны. Доля голосов остаётся рядом (``p_vote``), разброс ``p_std`` — сам по
себе сигнал трудности кейса и отдельная фича для стэка.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_reliability.methods.m3.axes import extract_axis_verdict

logger = logging.getLogger(__name__)

AGGREGATIONS: tuple[str, ...] = ("mean_prob", "vote_share", "median_prob")

# Версия одноосевого экстрактора: попадает в ключ кэша, как EXTRACTOR_VERSION
# у judge_client, чтобы старые записи не читались новым кодом молча.
AXIS_EXTRACTOR_VERSION = 1

_NO_VOTES_FALLBACK = 0.5


def _cache_path(cache_dir: Path, payload: Mapping[str, Any]) -> Path:
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return cache_dir / f"{hashlib.sha256(encoded).hexdigest()}.json"


def _cache_read(path: Path | None, n: int) -> list[dict] | None:
    """Сырые ответы модели из кэша; неполная или битая запись — промах."""
    if path is None or not path.exists():
        return None
    try:
        choices = json.loads(path.read_text(encoding="utf-8"))["choices"]
    except (json.JSONDecodeError, KeyError, TypeError, OSError):
        return None
    if not isinstance(choices, list) or len(choices) != n:
        return None
    required = ("text", "tokens", "finish_reason")
    if any(
        not isinstance(choice, dict) or any(key not in choice for key in required)
        for choice in choices
    ):
        return None
    return choices


def _cache_write(path: Path | None, choices: Sequence[dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"choices": list(choices)}, ensure_ascii=False), encoding="utf-8"
    )
    tmp.replace(path)  # atomic replace — прерывание не оставит битую запись


@dataclass(frozen=True)
class AxisAggregate:
    """Агрегат N сэмплов одной оси."""

    axis: str
    p: float
    p_vote: float
    p_std: float
    n: int
    n_voted: int


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def aggregate_axis(
    axis: str, probs: Sequence[float], votes: Sequence[int | None]
) -> AxisAggregate:
    """Среднее вероятностей, доля PASS-голосов и разброс по сэмплам.

    ``votes`` — дискретные вердикты из текста; ``None`` там, где вердикт не
    восстановился. Доля считается по восстановленным; если не восстановился ни
    один, доля равна 0.5 — та же конвенция, что у цепочки фолбэков.
    """
    if len(probs) != len(votes):
        raise ValueError(
            f"probs and votes must be aligned per sample, got {len(probs)} and {len(votes)}"
        )
    if not probs:
        raise ValueError(f"Cannot aggregate zero samples for axis {axis!r}")
    counted = [vote for vote in votes if vote is not None]
    return AxisAggregate(
        axis=axis,
        p=_mean(probs),
        p_vote=_mean(counted) if counted else _NO_VOTES_FALLBACK,
        p_std=float(statistics.pstdev(probs)) if len(probs) > 1 else 0.0,
        n=len(probs),
        n_voted=len(counted),
    )


def selfconsistency_scores(aggregate: AxisAggregate) -> dict[str, float]:
    """Скоры под именами из карточки C3: ``<axis>.p``, ``.p_vote``, ``.p_std``."""
    return {
        f"{aggregate.axis}.p": aggregate.p,
        f"{aggregate.axis}.p_vote": aggregate.p_vote,
        f"{aggregate.axis}.p_std": aggregate.p_std,
    }


def sample_axis(
    client: Any,
    system: str,
    user: str,
    *,
    axis: str,
    n: int = 8,
    temperature: float = 0.7,
    max_tokens: int = 800,
    top_p: float = 1.0,
    cache_dir: str | Path | None = None,
    cache_scope: str = "",
) -> list[tuple[float, dict]]:
    """N сэмплов одной оси -> список (вероятность PASS, meta) в порядке ответа.

    Отдельная функция, потому что абляции по N нужны СЫРЫЕ сэмплы: N=16
    сэмплируется один раз, а точки N=1,4,8 считаются по префиксам того же
    набора, без повторных вызовов модели.

    Кэшируются сырые ответы модели, а не вероятности: смена экстрактора не
    требует перегенерации. ``cache_scope`` описывает бэкенд (модель, endpoint),
    чтобы записи разных моделей не пересекались.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1 for axis {axis!r}, got {n}")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    path = (
        _cache_path(
            Path(cache_dir),
            {
                "axis": axis,
                "extractor_version": AXIS_EXTRACTOR_VERSION,
                "max_tokens": max_tokens,
                "n": n,
                "scope": cache_scope,
                "system": system,
                "temperature": temperature,
                "top_p": top_p,
                "user": user,
            },
        )
        if cache_dir is not None
        else None
    )
    cached = _cache_read(path, n)
    if cached is not None:
        return _verdicts(cached, axis)
    try:
        choices = client.chat(
            messages,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            top_p=top_p,
            logprobs=True,
        )
    except Exception as exc:  # noqa: BLE001 — HTTP-клиенты бросают свои типы ошибок
        # Провайдер без logprobs — деградируем в текстовый путь, как judge_client.
        logger.warning("Logprobs unavailable for axis %s; falling back to text: %s", axis, exc)
        choices = client.chat(
            messages,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            top_p=top_p,
            logprobs=False,
        )
    if len(choices) != n:
        raise RuntimeError(
            f"Judge returned {len(choices)} choice(s) for axis {axis!r} at n={n}: "
            "a sample would be silently dropped"
        )
    _cache_write(path, choices)
    return _verdicts(choices, axis)


def _verdicts(choices: Sequence[dict], axis: str) -> list[tuple[float, dict]]:
    return [
        extract_axis_verdict(
            choice["text"],
            choice["tokens"],
            axis,
            finish_reason=choice["finish_reason"],
        )
        for choice in choices
    ]


def judge_selfconsistent(
    client: Any,
    system: str,
    user: str,
    *,
    axis: str,
    n: int = 8,
    temperature: float = 0.7,
    max_tokens: int = 800,
    top_p: float = 1.0,
    cache_dir: str | Path | None = None,
    cache_scope: str = "",
) -> dict:
    """N сэмплов -> усреднение вероятностей + доля PASS-голосов.

    scores:
      ``<axis>.p``       среднее по сэмплам (усредняем ВЕРОЯТНОСТИ, не голоса:
                         голоса дают гранулярность 1/N, вероятности непрерывны)
      ``<axis>.p_vote``  доля сэмплов с PASS
      ``<axis>.p_std``   разброс — самостоятельный сигнал трудности кейса

    При ``n=1`` результат совпадает с одиночным вызовом судьи: ``<axis>.p`` равен
    вероятности единственного сэмпла, ``<axis>.p_std`` равен нулю.
    """
    samples = sample_axis(
        client,
        system,
        user,
        axis=axis,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        cache_dir=cache_dir,
        cache_scope=cache_scope,
    )
    probs = [probability for probability, _ in samples]
    metas = [meta for _, meta in samples]
    votes = [meta["verdict"] for meta in metas]
    aggregate = aggregate_axis(axis, probs, votes)
    return {
        **selfconsistency_scores(aggregate),
        "meta": {
            "axis": axis,
            "n": n,
            "temperature": temperature,
            "probs": probs,
            "votes": votes,
            "methods": [str(meta["method"]) for meta in metas],
            "markers": [meta["marker"] for meta in metas],
            "n_voted": aggregate.n_voted,
            "raw": str(metas[0]["raw"]),
        },
    }


# --- абляция качество/цена по N и T -----------------------------------------


@dataclass(frozen=True)
class AblationRecord:
    """Сырые сэмплы одного кейса по одной оси плюс золотая метка."""

    case_id: str
    probs: tuple[float, ...]
    votes: tuple[int | None, ...]
    gold: int


def aggregate_prefix(
    probs: Sequence[float], votes: Sequence[int | None], n: int, aggregation: str
) -> float:
    """Агрегат первых ``n`` сэмплов — точка сетки абляции без новых вызовов."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation {aggregation!r}, expected one of {AGGREGATIONS}")
    if n > len(probs):
        raise ValueError(f"Requested n={n} but only {len(probs)} sample(s) are available")
    head_probs = list(probs[:n])
    head_votes = [vote for vote in votes[:n] if vote is not None]
    if aggregation == "mean_prob":
        return _mean(head_probs)
    if aggregation == "median_prob":
        return float(statistics.median(head_probs))
    return _mean(head_votes) if head_votes else _NO_VOTES_FALLBACK


def _auc_with_ci(
    gold: Sequence[int], scores: Sequence[float], *, seed: int, replicates: int
) -> tuple[float, tuple[float, float]]:
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    from rag_reliability.evaluation.bootstrap import bootstrap_ci  # noqa: PLC0415

    if len(set(gold)) < 2:
        raise ValueError(
            f"AUC needs both classes among {len(gold)} case(s); gold has only {set(gold)}"
        )

    def metric(y: Any, pred: Any) -> float:
        # Резэмплинг может выдать одноклассовый фолд — там AUC не определён,
        # 0.5 = «ранжирования нет», интервал от этого только шире.
        return 0.5 if len(set(y.tolist())) < 2 else float(roc_auc_score(y, pred))

    result = bootstrap_ci(list(gold), list(scores), metric, B=replicates, seed=seed)
    return result.point, (result.lo, result.hi)


def build_ablation_table(
    records: Sequence[AblationRecord],
    *,
    axis: str,
    temperature: float,
    ns: Sequence[int],
    aggregations: Sequence[str] = AGGREGATIONS,
    seed: int = 0,
    replicates: int = 2000,
    auc_fn: Callable[[Sequence[int], Sequence[float]], tuple[float, tuple[float, float]]]
    | None = None,
) -> list[dict]:
    """Таблица качество/цена: AUC с 95% ДИ и цена в сэмплах на кейс.

    Требуется контрактом платформы и не строилась ни разу. Цена измеряется в
    сэмплах на кейс: ровно то, что растёт линейно по N.
    """
    if not records:
        raise ValueError("Cannot build an ablation table from zero records")
    grid = sorted({int(value) for value in ns})
    if not grid or grid[0] < 1:
        raise ValueError(f"ns must be positive integers, got {list(ns)}")
    short = [
        record.case_id for record in records if len(record.probs) < grid[-1]
    ]
    if short:
        raise ValueError(
            f"{len(short)} case(s) carry fewer than N={grid[-1]} samples, e.g. {short[:5]}"
        )
    gold = [record.gold for record in records]
    measure = auc_fn or (
        lambda y, scores: _auc_with_ci(y, scores, seed=seed, replicates=replicates)
    )

    rows: list[dict] = []
    for n in grid:
        for aggregation in aggregations:
            scores = [
                aggregate_prefix(record.probs, record.votes, n, aggregation)
                for record in records
            ]
            auc, ci95 = measure(gold, scores)
            rows.append(
                {
                    "axis": axis,
                    "temperature": float(temperature),
                    "n": n,
                    "aggregation": aggregation,
                    "auc": float(auc),
                    "ci95": [float(ci95[0]), float(ci95[1])],
                    "n_cases": len(records),
                    "pass_rate": _mean([float(score >= 0.5) for score in scores]),
                    "frac_above_099": _mean([float(score > 0.99) for score in scores]),
                    "cost_samples_per_case": n,
                    "cost_relative": n / grid[0],
                }
            )
    return rows


def render_ablation_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Таблица для отчёта и описания PR."""
    header = (
        "| ось | T | N | агрегация | AUC [95% ДИ] | доля PASS | доля p>0.99 | "
        "цена, сэмплов/кейс |"
    )
    separator = "|---|---:|---:|---|---|---:|---:|---:|"
    lines = [header, separator]
    for row in rows:
        ci_lo, ci_hi = row["ci95"]
        lines.append(
            f"| {row['axis']} | {row['temperature']:.1f} | {row['n']} | {row['aggregation']} "
            f"| {row['auc']:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] "
            f"| {row['pass_rate']:.1%} | {row['frac_above_099']:.1%} "
            f"| {row['cost_samples_per_case']} |"
        )
    return "\n".join(lines)
