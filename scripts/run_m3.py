#!/usr/bin/env python
"""Run Method 3 prompt judge through the shared predictions contract.

Two prompt styles:

* ``joint``  — исторический путь: один вызов, оба вердикта в одном ответе;
* ``axes``   — два независимых вызова (faithfulness и relevance), промпты из
  ``configs/prompts/*.yaml``, опционально self-consistency по N сэмплам.

Абляция ``--ablation-n`` считает таблицу качество/цена по N и T, переиспользуя
один набор сэмплов N=max: точки сетки берутся по префиксам.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable
from typing import Any

from tqdm import tqdm

from rag_reliability.dataset import load_jsonl, save_jsonl
from rag_reliability.dummy_model import STRATEGIES, DummyPredictor
from rag_reliability.methods.m3 import build_system_prompt, build_user_prompt, parse_m3_prediction
from rag_reliability.methods.m3.axes import (
    AXES,
    AXIS_FAITHFULNESS,
    AXIS_RELEVANCE,
    axis_anchor,
    build_axis_prompt,
    prompt_versions,
)
from rag_reliability.methods.m3.selfconsistency import (
    AblationRecord,
    build_ablation_table,
    judge_selfconsistent,
    render_ablation_markdown,
    sample_axis,
)
from rag_reliability.mlx_backend import make_generate_fn
from rag_reliability.parsing import parse_prediction
from rag_reliability.schema import Prediction

AXIS_SCORE_KEYS = {AXIS_FAITHFULNESS: "p_faith", AXIS_RELEVANCE: "p_rel"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl")
    parser.add_argument("--output", default="results/m3_zero_shot_predictions.jsonl")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "gepa"], default="zero_shot")
    parser.add_argument("--examples", default=None, help="YAML examples for --mode few_shot")
    parser.add_argument("--prompt-file", default=None, help="Prompt text for --mode gepa")
    parser.add_argument(
        "--backend", choices=["dummy", "mlx", "openai", "openai_judge"], default="mlx"
    )
    parser.add_argument("--dummy-strategy", choices=list(STRATEGIES), default="always_reliable")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--run-meta", default=None)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="parallel requests for the openai_judge backend (1 = sync client)",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["joint", "axes", "perchunk"],
        default="axes",
        help=(
            "axes (по умолчанию) = отдельный вызов на ось; joint = прежний общий вызов; "
            "perchunk = по вызову на чанк, только ось faithfulness"
        ),
    )
    parser.add_argument("--prompts-dir", default="configs/prompts")
    parser.add_argument("--prompt-file-faithfulness", default=None, help="--mode gepa, ось faith")
    parser.add_argument("--prompt-file-relevance", default=None, help="--mode gepa, ось relevance")
    parser.add_argument(
        "--sc-n", type=int, default=1, help="сэмплов на ось (self-consistency); 1 = один вызов"
    )
    parser.add_argument(
        "--sc-temperature",
        type=float,
        default=0.0,
        help="температура сэмплирования; для --sc-n > 1 осмысленны 0.7-1.0",
    )
    parser.add_argument(
        "--ablation-n",
        default=None,
        help="сетка N через запятую (например 1,4,8,16) — включает абляционный прогон",
    )
    parser.add_argument("--ablation-temperature", default="0.7,1.0", help="сетка T через запятую")
    parser.add_argument("--ablation-out", default=None, help="куда писать таблицу качество/цена")
    parser.add_argument(
        # %% — argparse прогоняет help через %-форматирование.
        "--ablation-replicates", type=int, default=2000, help="реплик бутстрэпа для 95%% ДИ"
    )
    return parser.parse_args()


def _flag(args: argparse.Namespace, name: str, default: Any) -> Any:
    """Мягкое чтение новых флагов.

    ``main()`` вызывают не только из CLI, но и с Namespace, собранным вручную
    (tests/test_m3_method.py), поэтому отсутствие нового флага означает значение
    по умолчанию, а не ошибку. Значения совпадают с argparse, чтобы у CLI и у
    ручного вызова не расходилось поведение. К фичам и вероятностям это
    послабление не относится.
    """
    return getattr(args, name, default)


def _prob_prediction(sample_id: str, p_faith: float, p_rel: float, meta: dict) -> Prediction:
    """Probabilities -> binary contract fields at 0.5, probabilities kept alongside."""
    return Prediction(
        id=sample_id,
        faithfulness_pred=int(p_faith >= 0.5),
        relevance_pred=int(p_rel >= 0.5),
        raw_output=str(meta.get("raw", "")),
        invalid_output=meta.get("method") == "default",
        faithfulness_prob=p_faith,
        relevance_prob=p_rel,
        prob_method=str(meta.get("method", "")),
    )


def run_openai_judge(args: argparse.Namespace, samples: list) -> list[Prediction]:
    """Logprob-probability judge over an OpenAI-compatible endpoint (sync or async)."""
    from rag_reliability.methods.m3.judge_client import (  # noqa: PLC0415
        AsyncJudgeClient,
        JudgeClient,
    )

    system_prompt = build_system_prompt(
        args.mode, examples_path=args.examples, prompt_file=args.prompt_file
    )
    common = {
        "model": args.model,
        "api_base": args.api_base,
        "api_key": os.environ.get(args.api_key_env, ""),
    }
    users = [build_user_prompt(s, args.max_context_chars) for s in samples]

    if args.concurrency > 1:
        import asyncio  # noqa: PLC0415

        client = AsyncJudgeClient(
            cache_dir=args.cache_dir, concurrency=args.concurrency, **common
        )
        results = asyncio.run(client.judge_many(system_prompt, users, max_tokens=args.max_tokens))
    else:
        sync_client = JudgeClient(cache_dir=args.cache_dir, **common)
        results = [
            sync_client.judge(system_prompt, user, max_tokens=args.max_tokens)
            for user in tqdm(users, desc="m3/openai_judge")
        ]
    return [
        _prob_prediction(sample.id, p_f, p_r, meta)
        for sample, (p_f, p_r, meta) in zip(samples, results, strict=True)
    ]


class DummyAxisClient:
    """Стенд без сети: вердикт и логпробы выводятся из хеша промпта.

    Нужен ровно для смоука одноосевого пайплайна (тесты, `--limit`), а не как
    бейзлайн: числа детерминированы, но никакого отношения к качеству не имеют.
    Сэмплы при n>1 различаются, поэтому p_std ведёт себя как на реальной модели.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.calls: list[dict] = []

    @staticmethod
    def _unit(*parts: str) -> float:
        digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
        return (int(digest[:8], 16) + 0.5) / float(1 << 32)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: int = 512,
        top_p: float = 1.0,
        logprobs: bool = False,
    ) -> list[dict]:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        anchor = next(
            (axis_anchor(axis) for axis in AXES if axis_anchor(axis) in system),
            axis_anchor(AXIS_FAITHFULNESS),
        )
        self.calls.append({"n": n, "temperature": temperature, "anchor": anchor})
        choices = []
        for index in range(n):
            jitter = (self._unit(user, str(index)) - 0.5) * temperature
            probability = min(max(self._unit(system[:64], user) + jitter, 0.01), 0.99)
            verdict = "PASS" if probability >= 0.5 else "FAIL"
            text = f"ANALYSIS: смоук\nMARKER: none\n{anchor}: {verdict}"
            tokens = (
                [
                    {"token": anchor, "logprob": -0.1, "top": {}},
                    {"token": ":", "logprob": -0.1, "top": {}},
                    {
                        "token": f" {verdict}",
                        "logprob": math.log(max(probability, 1e-9)),
                        "top": {
                            " PASS": math.log(probability),
                            " FAIL": math.log(1.0 - probability),
                        },
                    },
                ]
                if logprobs
                else []
            )
            choices.append({"text": text, "tokens": tokens, "finish_reason": "stop"})
        return choices


class TextAxisClient:
    """Адаптер текстовых бэкендов (mlx, openai) под контракт ``chat()``.

    Логпробов эти бэкенды не отдают, поэтому вероятность придёт из regex-ветки
    цепочки (0.9/0.1) — грубее, чем logprobs, но ось не теряется и кейс не
    выпадает. Нужен, чтобы одноосевой путь был доступен на всех бэкендах, на
    которых работал общий, а не только на openai_judge.
    """

    def __init__(self, generate: Callable[[str, str, int, float], str]) -> None:
        self._generate = generate

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: int = 512,
        top_p: float = 1.0,
        logprobs: bool = False,
    ) -> list[dict]:
        system, user = messages[0]["content"], messages[-1]["content"]
        return [
            {
                "text": self._generate(system, user, max_tokens, temperature),
                "tokens": [],
                "finish_reason": "stop",
            }
            for _ in range(n)
        ]


def build_axis_client(args: argparse.Namespace) -> Any:
    """Клиент для одноосевого пути: судья с логпробами, текстовый бэкенд или стенд."""
    if args.backend == "dummy":
        return DummyAxisClient()
    if args.backend == "openai_judge":
        from rag_reliability.methods.m3.judge_client import JudgeClient  # noqa: PLC0415

        return JudgeClient(
            model=args.model,
            api_base=args.api_base,
            api_key=os.environ.get(args.api_key_env, ""),
            cache_dir=args.cache_dir,
        )
    if args.backend == "openai":
        from rag_reliability.methods.m3.openai_client import CachedChatClient  # noqa: PLC0415

        chat_client = CachedChatClient(
            model=args.model,
            api_base=args.api_base,
            api_key=os.environ.get(args.api_key_env, ""),
            cache_dir=args.cache_dir,
        )
        return TextAxisClient(
            lambda system, user, max_tokens, temperature: chat_client.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
    if args.backend == "mlx":
        generate_fn = make_generate_fn(args.model, args.max_tokens)
        return TextAxisClient(
            lambda system, user, max_tokens, temperature: generate_fn(f"{system}\n\n{user}")
        )
    raise ValueError(
        f"--prompt-style axes has no client for backend {args.backend!r}; "
        "expected one of 'openai_judge', 'openai', 'mlx', 'dummy'"
    )


def _cache_scope(args: argparse.Namespace) -> str:
    """Идентичность бэкенда в ключе кэша одноосевых вызовов."""
    return f"{args.model}|{args.api_base}|{args.backend}"


def _axis_prompt_file(args: argparse.Namespace, axis: str) -> str | None:
    return _flag(args, f"prompt_file_{axis}", None)


def _axis_prompts(args: argparse.Namespace, sample: Any, axis: str) -> tuple[str, str]:
    return build_axis_prompt(
        sample,
        axis,
        mode=args.mode,
        examples=args.examples,
        prompts_dir=_flag(args, "prompts_dir", None),
        prompt_file=_axis_prompt_file(args, axis),
        max_context_chars=args.max_context_chars,
    )


def _pick_marker(results: dict[str, dict]) -> str:
    """Код маркера самой уверенной провалившейся оси; иначе none."""
    failing = [
        (results[axis][f"{axis}.p"], axis) for axis in AXES if results[axis][f"{axis}.p"] < 0.5
    ]
    if not failing:
        return "none"
    _, axis = min(failing)
    markers = [marker for marker in results[axis]["meta"]["markers"] if marker]
    if not markers:
        return "unknown"
    return Counter(markers).most_common(1)[0][0]


def _axes_prediction(
    sample_id: str, results: dict[str, dict], versions: dict[str, str]
) -> Prediction:
    """Две оси -> одна строка предсказаний общего контракта."""
    p_faith = results[AXIS_FAITHFULNESS][f"{AXIS_FAITHFULNESS}.p"]
    p_rel = results[AXIS_RELEVANCE][f"{AXIS_RELEVANCE}.p"]
    methods = {axis: results[axis]["meta"]["methods"][0] for axis in AXES}
    scores: dict[str, float] = {}
    for axis in AXES:
        key = AXIS_SCORE_KEYS[axis]
        scores[f"m3.{key}"] = float(results[axis][f"{axis}.p"])
        scores[f"m3.{key}_vote"] = float(results[axis][f"{axis}.p_vote"])
        scores[f"m3.{key}_std"] = float(results[axis][f"{axis}.p_std"])
    raw = {
        "prompt_versions": versions,
        **{
            axis: {
                "raw": results[axis]["meta"]["raw"],
                "n": results[axis]["meta"]["n"],
                "methods": results[axis]["meta"]["methods"],
            }
            for axis in AXES
        },
    }
    return Prediction(
        id=sample_id,
        faithfulness_pred=int(p_faith >= 0.5),
        relevance_pred=int(p_rel >= 0.5),
        marker_pred=_pick_marker(results),
        raw_output=json.dumps(raw, ensure_ascii=False),
        invalid_output=all(method == "default" for method in methods.values()),
        faithfulness_prob=p_faith,
        relevance_prob=p_rel,
        prob_method="+".join(sorted(set(methods.values()))),
        scores=scores,
    )


def run_axes(args: argparse.Namespace, samples: list) -> list[Prediction]:
    """Две оси в двух вызовах; при --sc-n > 1 каждая ось усредняется по сэмплам."""
    client = build_axis_client(args)
    versions = prompt_versions(prompts_dir=_flag(args, "prompts_dir", None))
    n = int(_flag(args, "sc_n", 1))
    temperature = float(_flag(args, "sc_temperature", 0.0))

    predictions: list[Prediction] = []
    for sample in tqdm(samples, desc=f"m3/axes/{args.backend}"):
        results = {}
        for axis in AXES:
            system, user = _axis_prompts(args, sample, axis)
            results[axis] = judge_selfconsistent(
                client,
                system,
                user,
                axis=axis,
                n=n,
                temperature=temperature,
                max_tokens=args.max_tokens,
                cache_dir=args.cache_dir,
                cache_scope=_cache_scope(args),
            )
        predictions.append(_axes_prediction(sample.id, results, versions))
    return predictions


def build_perchunk_client(args: argparse.Namespace) -> Any:
    """Клиент пофрагментного пути: асинхронный при --concurrency > 1.

    Пофрагментная верификация шлёт по запросу на чанк, то есть 5-15 запросов на
    кейс вместо двух. Синхронный клиент делает их по очереди, и корпусный
    прогон растягивается на часы там, где узкое место — сеть, а не GPU.
    """
    if args.backend == "dummy":
        return DummyAxisClient()
    if args.backend not in ("openai_judge", "openai"):
        raise ValueError(
            f"--prompt-style perchunk has no client for backend {args.backend!r}; "
            "expected 'openai_judge' or 'dummy' (per-chunk scoring reads verdict logprobs)"
        )
    from rag_reliability.methods.m3.judge_client import (  # noqa: PLC0415
        AsyncJudgeClient,
        JudgeClient,
    )

    common = {
        "model": args.model,
        "api_base": args.api_base,
        "api_key": os.environ.get(args.api_key_env, ""),
        "cache_dir": args.cache_dir,
    }
    concurrency = int(_flag(args, "concurrency", 1))
    if concurrency > 1:
        return AsyncJudgeClient(concurrency=concurrency, **common)
    return JudgeClient(**common)


def run_perchunk(args: argparse.Namespace, samples: list) -> list[Prediction]:
    """Пофрагментная верификация faithfulness: по вызову на каждый чанк.

    Ось relevance чанки не получает по контракту C3 — её определение опирается
    только на вопрос и ответ, — поэтому вероятностей осей здесь нет вовсе.
    Артефакт несёт пять фич, а агрегат из них выбирает стэкер.
    """
    from rag_reliability.methods.m3.perchunk import score_per_chunk  # noqa: PLC0415

    client = build_perchunk_client(args)
    predictions: list[Prediction] = []
    for sample in tqdm(samples, desc=f"m3/perchunk/{args.backend}"):
        features = score_per_chunk(sample, client)
        predictions.append(
            Prediction(
                id=sample.id,
                faithfulness_pred=0,
                relevance_pred=0,
                prob_method="perchunk_logprobs",
                scores={key: float(value) for key, value in features.items()},
            )
        )
    return predictions


def _grid(raw: str, cast: Any) -> list:
    values = [chunk.strip() for chunk in str(raw).split(",") if chunk.strip()]
    if not values:
        raise ValueError(f"Empty grid: {raw!r}")
    return [cast(value) for value in values]


def run_ablation(args: argparse.Namespace, samples: list) -> dict:
    """Таблица качество/цена по N и T.

    Сэмплируем один раз с N=max и берём точки сетки по префиксам: сетка
    {1,4,8,16} стоит ровно один прогон N=16.

    Использует золотые метки, поэтому её место — val/train-часть протокола,
    а не сплит, по которому отчитываются.
    """
    client = build_axis_client(args)
    ns = _grid(_flag(args, "ablation_n", ""), int)
    temperatures = _grid(_flag(args, "ablation_temperature", "0.7,1.0"), float)
    n_max = max(ns)
    gold = {
        AXIS_FAITHFULNESS: lambda s: s.faithfulness,
        AXIS_RELEVANCE: lambda s: s.relevance,
    }

    rows: list[dict] = []
    for temperature in temperatures:
        for axis in AXES:
            records = []
            for sample in tqdm(samples, desc=f"ablation/{axis}/T={temperature}"):
                system, user = _axis_prompts(args, sample, axis)
                drawn = sample_axis(
                    client,
                    system,
                    user,
                    axis=axis,
                    n=n_max,
                    temperature=temperature,
                    max_tokens=args.max_tokens,
                    cache_dir=args.cache_dir,
                    cache_scope=_cache_scope(args),
                )
                records.append(
                    AblationRecord(
                        case_id=sample.id,
                        probs=tuple(probability for probability, _ in drawn),
                        votes=tuple(meta["verdict"] for _, meta in drawn),
                        gold=int(gold[axis](sample)),
                    )
                )
            rows += build_ablation_table(
                records,
                axis=axis,
                temperature=temperature,
                ns=ns,
                replicates=int(_flag(args, "ablation_replicates", 2000)),
            )
    return {
        "config": {
            "ns": ns,
            "temperatures": temperatures,
            "model": args.model,
            "backend": args.backend,
            "n_cases": len(samples),
            "prompt_versions": prompt_versions(prompts_dir=_flag(args, "prompts_dir", None)),
        },
        "rows": rows,
        "markdown": render_ablation_markdown(rows),
    }


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]

    if _flag(args, "ablation_n", None):
        report = run_ablation(args, samples)
        print(report["markdown"])
        destination = _flag(args, "ablation_out", None)
        if destination:
            with open(destination, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
            print(f"Wrote ablation table to {destination}")
        if args.run_meta:
            from rag_reliability.run_meta import write_run_meta  # noqa: PLC0415

            write_run_meta(args.run_meta, args)
        return

    prompt_style = _flag(args, "prompt_style", "axes")
    if prompt_style == "perchunk":
        predictions = run_perchunk(args, samples)
        save_jsonl(predictions, args.output)
        print(f"Wrote {len(predictions)} per-chunk prediction(s) to {args.output}")
        if args.run_meta:
            from rag_reliability.run_meta import write_run_meta  # noqa: PLC0415

            write_run_meta(args.run_meta, args)
        return

    if prompt_style == "axes":
        predictions = run_axes(args, samples)
        save_jsonl(predictions, args.output)
        invalid = sum(prediction.invalid_output for prediction in predictions)
        print(f"Wrote {len(predictions)} predictions to {args.output} (invalid outputs: {invalid})")
        if args.run_meta:
            from rag_reliability.run_meta import write_run_meta  # noqa: PLC0415

            args.prompt_versions = prompt_versions(prompts_dir=_flag(args, "prompts_dir", None))
            write_run_meta(args.run_meta, args)
        return

    if args.backend == "openai_judge":
        predictions = run_openai_judge(args, samples)
        save_jsonl(predictions, args.output)
        invalid = sum(prediction.invalid_output for prediction in predictions)
        print(f"Wrote {len(predictions)} predictions to {args.output} (invalid outputs: {invalid})")
        if args.run_meta:
            from rag_reliability.run_meta import write_run_meta  # noqa: PLC0415

            write_run_meta(args.run_meta, args)
        return

    if args.backend == "dummy":
        predictor = DummyPredictor(strategy=args.dummy_strategy, mode="direct")
        generate_fn = None
        chat_client = None
    elif args.backend == "openai":
        from rag_reliability.methods.m3.openai_client import CachedChatClient  # noqa: PLC0415

        predictor = None
        generate_fn = None
        chat_client = CachedChatClient(
            model=args.model,
            api_base=args.api_base,
            api_key=os.environ.get(args.api_key_env, ""),
            cache_dir=args.cache_dir,
        )
    else:
        predictor = None
        generate_fn = make_generate_fn(args.model, args.max_tokens)
        chat_client = None

    system_prompt = build_system_prompt(
        args.mode,
        examples_path=args.examples,
        prompt_file=args.prompt_file,
    )

    predictions: list[Prediction] = []
    for sample in tqdm(samples, desc=f"m3/{args.backend}"):
        if predictor is not None:
            raw_output = predictor.predict(sample)
            prediction = parse_prediction(raw_output, sample.id)
        else:
            user_prompt = build_user_prompt(sample, args.max_context_chars)
            if chat_client is not None:
                raw_output = chat_client.chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=args.max_tokens,
                )
            else:
                prompt = f"{system_prompt}\n\n{user_prompt}"
                raw_output = generate_fn(prompt)
            prediction = parse_m3_prediction(raw_output, sample.id)
        predictions.append(prediction)

    save_jsonl(predictions, args.output)
    invalid = sum(prediction.invalid_output for prediction in predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output} (invalid outputs: {invalid})")
    if args.run_meta:
        from rag_reliability.run_meta import write_run_meta  # noqa: PLC0415

        write_run_meta(args.run_meta, args)


if __name__ == "__main__":
    main()
