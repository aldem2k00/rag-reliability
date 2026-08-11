"""Пофрагментная верификация faithfulness для Метода 3.

Каждый запрос видит ровно один retrieved-чанк. Все запросы одного кейса
передаются batch-функции одновременно: так вызывающая сторона может отправить
их через асинхронный клиент с семафором, не делая последовательный цикл сети.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from rag_reliability.methods.m3.axes import (
    AXIS_FAITHFULNESS,
    build_axis_prompt,
    extract_axis_verdict,
)
from rag_reliability.methods.surface.features import split_chunks
from rag_reliability.schema import RagSample

BatchJudgeResult = Sequence[float] | Awaitable[Sequence[float]]
BatchJudgeFn = Callable[[str, Sequence[str]], BatchJudgeResult]

SUPPORT_THRESHOLD = 0.5
_MAX_TOKENS = 800
_CACHE_VERSION = 1
logger = logging.getLogger(__name__)


class ChatClient(Protocol):
    """Минимальная общая поверхность sync/async M3-клиентов."""

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Sequence[dict] | Awaitable[Sequence[dict]]: ...


def _chunk_prompts(sample: RagSample, axis: str) -> tuple[str, list[str]]:
    """Собрать общий system и по одному изолированному user-промпту на чанк."""
    chunks = split_chunks(sample.context)
    if not chunks:
        raise ValueError(f"Sample {sample.id!r} has no context chunks for per-chunk scoring")

    system: str | None = None
    users: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_sample = sample.model_copy(update={"context": f"[CHUNK {index}]\n{chunk}"})
        chunk_system, user = build_axis_prompt(chunk_sample, axis)
        if system is None:
            system = chunk_system
        elif chunk_system != system:
            raise RuntimeError(
                f"Faithfulness system prompt changed within sample {sample.id!r} at chunk {index}"
            )
        users.append(user)

    if system is None:  # pragma: no cover — непустота chunks проверена выше
        raise RuntimeError(f"Failed to build per-chunk prompts for sample {sample.id!r}")
    return system, users


def _validated_scores(
    raw_scores: Sequence[float], *, sample_id: str, expected: int
) -> list[float]:
    scores = [float(score) for score in raw_scores]
    if len(scores) != expected:
        raise ValueError(
            f"Batch judge returned {len(scores)} score(s) for sample {sample_id!r}; "
            f"expected one score for each of {expected} chunk(s)"
        )
    invalid = [score for score in scores if not math.isfinite(score) or not 0.0 <= score <= 1.0]
    if invalid:
        raise ValueError(
            f"Batch judge returned invalid probability score(s) for sample {sample_id!r}: "
            f"{invalid[:5]}"
        )
    return scores


async def _await_result(result: Awaitable[Sequence[float]]) -> Sequence[float]:
    return await result


def _resolve_batch(result: BatchJudgeResult, *, sample_id: str) -> Sequence[float]:
    """Синхронный фасад принимает и обычный, и async batch-адаптер."""
    if not inspect.isawaitable(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_result(result))
    if inspect.iscoroutine(result):
        result.close()
    raise RuntimeError(
        f"Async batch judge for sample {sample_id!r} cannot be awaited from a running event loop; "
        "call score_per_chunk from synchronous orchestration"
    )


async def _chat(
    client: ChatClient,
    messages: list[dict[str, str]],
    *,
    logprobs: bool,
) -> Sequence[dict]:
    kwargs = {
        "temperature": 0.0,
        "n": 1,
        "max_tokens": _MAX_TOKENS,
        "top_p": 1.0,
        "logprobs": logprobs,
    }
    if inspect.iscoroutinefunction(client.chat):
        result = client.chat(messages, **kwargs)
        if not inspect.isawaitable(result):
            raise TypeError("Async M3 client.chat returned a non-awaitable result")
        return await result
    return await asyncio.to_thread(client.chat, messages, **kwargs)


def _cache_path(client: ChatClient, system: str, user: str, axis: str) -> Path | None:
    cache_dir = getattr(client, "cache_dir", None)
    if cache_dir is None:
        return None
    model = getattr(client, "model", None)
    api_base = getattr(client, "api_base", None)
    if not isinstance(model, str) or not isinstance(api_base, str):
        raise ValueError(
            "M3 chat client with cache_dir must expose string model and api_base attributes"
        )
    payload = {
        "api_base": api_base,
        "axis": axis,
        "cache_version": _CACHE_VERSION,
        "max_tokens": _MAX_TOKENS,
        "model": model,
        "system": system,
        "temperature": 0.0,
        "top_p": 1.0,
        "user": user,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"perchunk-{hashlib.sha256(encoded).hexdigest()}.json"


def _read_choice(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        choice = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(choice, dict):
        return None
    required = ("text", "tokens", "finish_reason")
    if any(key not in choice for key in required):
        return None
    if not isinstance(choice["text"], str) or not isinstance(choice["tokens"], list):
        return None
    return choice


def _write_choice(path: Path | None, choice: dict) -> None:
    if path is None:
        return
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(choice, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


async def _score_chat_prompt(
    client: ChatClient,
    system: str,
    user: str,
    *,
    axis: str,
    sem: asyncio.Semaphore,
) -> float:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    path = _cache_path(client, system, user, axis)
    choice = _read_choice(path)
    if choice is None:
        async with sem:
            try:
                choices = await _chat(client, messages, logprobs=True)
            except Exception as exc:  # noqa: BLE001 — HTTP-клиенты бросают provider-specific типы
                logger.warning(
                    "Logprobs unavailable for per-chunk axis %s; falling back to text: %s",
                    axis,
                    exc,
                )
                choices = await _chat(client, messages, logprobs=False)
        if len(choices) != 1:
            raise RuntimeError(
                f"M3 chat client returned {len(choices)} choice(s) for one per-chunk prompt; "
                "expected exactly 1"
            )
        choice = dict(choices[0])
        _write_choice(path, choice)
    missing = [key for key in ("text", "tokens", "finish_reason") if key not in choice]
    if missing:
        raise ValueError(f"M3 chat choice is missing required key(s): {missing}")
    probability, _ = extract_axis_verdict(
        str(choice["text"]),
        choice["tokens"],
        axis,
        finish_reason=choice["finish_reason"],
    )
    return probability


async def _score_chat_client(
    client: ChatClient,
    system: str,
    users: Sequence[str],
    *,
    axis: str,
) -> list[float]:
    """Все чанки запускаются одной ``gather``-пачкой с семафором клиента."""
    is_async = inspect.iscoroutinefunction(client.chat)
    concurrency = getattr(client, "concurrency", len(users) if is_async else 1)
    if not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError(f"M3 chat client concurrency must be a positive integer, got {concurrency!r}")
    sem = asyncio.Semaphore(concurrency)
    return list(
        await asyncio.gather(
            *[
                _score_chat_prompt(client, system, user, axis=axis, sem=sem)
                for user in users
            ]
        )
    )


def _judge_scores(
    judge: BatchJudgeFn | ChatClient,
    system: str,
    users: Sequence[str],
    *,
    axis: str,
    sample_id: str,
) -> Sequence[float]:
    if callable(judge):
        return _resolve_batch(judge(system, users), sample_id=sample_id)
    if not hasattr(judge, "chat"):
        raise TypeError(
            "judge_fn must be a batch callable or an M3 client exposing chat(messages, ...)"
        )
    return _resolve_batch(
        _score_chat_client(judge, system, users, axis=axis),
        sample_id=sample_id,
    )


def score_per_chunk(
    sample: RagSample,
    judge_fn: BatchJudgeFn | ChatClient,
    *,
    axis: str = AXIS_FAITHFULNESS,
) -> dict[str, float]:
    """Оценить ответ отдельно по каждому чанку и вернуть пять ``m3.*`` фич.

    Batch-callback получает общий system и все user-промпты одним вызовом.
    ``JudgeClient``/``AsyncJudgeClient`` можно передать напрямую: их raw choices
    обрабатываются одноосевой цепочкой C3, а запросы запускаются через gather с
    семафором. Relevance по контракту C3 чанки не получает.
    """
    if axis != AXIS_FAITHFULNESS:
        raise ValueError(
            f"Per-chunk scoring supports only axis={AXIS_FAITHFULNESS!r}, got {axis!r}"
        )

    system, users = _chunk_prompts(sample, axis)
    scores = _validated_scores(
        _judge_scores(
            judge_fn,
            system,
            users,
            axis=axis,
            sample_id=sample.id,
        ),
        sample_id=sample.id,
        expected=len(users),
    )

    ranked = sorted(scores, reverse=True)
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else top1
    argmax = max(range(len(scores)), key=scores.__getitem__) + 1
    return {
        "m3.max_chunk_score": top1,
        "m3.mean_chunk_score": math.fsum(scores) / len(scores),
        "m3.chunk_disagreement": top1 - top2,
        "m3.n_supporting": float(sum(score > SUPPORT_THRESHOLD for score in scores)),
        "m3.argmax_chunk": float(argmax),
    }
