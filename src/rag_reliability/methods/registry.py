# src/rag_reliability/methods/registry.py
"""Single source of truth for reliability methods: metadata + CLI command builders.

Реестр несёт два независимых интерфейса метода:

* ``build_command`` — как запустить метод отдельным процессом. Им пользуются
  ``scripts/run_benchmark.py``, ``rag-judge`` и демо; трогать его нельзя.
* ``build_scorer`` — как посчитать один кейс в текущем процессе. На нём стоит
  ``scripts/score.py``: продолжение прерванного прогона (``--resume``) и
  инкрементальная запись невозможны, если метод виден только как subprocess.

Плюс контракт артефакта: ``score_keys`` перечисляет ключи ``Prediction.scores``,
которые метод обязан заполнить. Он проверяется после каждого прогона
(``validate_scores_file``) — иначе молчаливая деградация метода выглядит как
успешный прогон.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_reliability.methods.surface.features import FEATURE_KEYS

if TYPE_CHECKING:
    from rag_reliability.schema import Prediction, RagSample

# Префиксы ключей scores, закреплённые за методами (HANDOFF.md §7.1 + карточка B2).
# prompt./lora. добавлены под семейства реестра, которых §7.1 не знала: свой префикс
# на семейство — единственное, что гарантирует отсутствие коллизий в стэкере.
SCORE_PREFIXES: tuple[str, ...] = (
    "surf.", "m3.", "m6.", "enc.", "ld.", "ind.", "prompt.", "lora.", "stack.",
)

# Дамми-бэкенды существуют ради смоуков пайплайна, а не ради сигнала.
DUMMY_METHODS: frozenset[str] = frozenset({"dummy_direct", "dummy_marker"})


@dataclass(frozen=True)
class CommandContext:
    data: Path
    run_dir: Path
    predictions_path: Path
    python: str = "python"
    model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    max_tokens: int = 64
    direct_adapter_path: str = "results/adapters_direct"
    marker_adapter_path: str = "results/adapters_marker"
    lettucedetect_model: str = "results/lettucedetect/classifier.joblib"
    encoder_model: str = "deepvk/RuModernBERT-base"
    encoder_output_dir: str | None = None
    encoder_max_length: int = 512
    encoder_batch_size: int = 4
    encoder_epochs: float = 3
    encoder_learning_rate: float = 2e-5
    encoder_pos_weight_mode: str = "none"
    m3_backend: str = "mlx"
    m3_max_tokens: int = 400
    m3_max_context_chars: int | None = None
    m3_examples: str = "configs/few_shot.yaml"
    m3_prompt_file: str = "configs/m3_gepa_prompt.txt"
    m3_api_base: str = "http://localhost:8000/v1"
    m3_api_key_env: str = "OPENAI_API_KEY"
    m3_cache_dir: str = "results/m3/cache"
    m3_concurrency: int = 1
    m3_dummy_strategy: str = "always_reliable"
    m6_features: str = "results/m6/features.jsonl"
    m6_backend: str = "dummy"
    m6_samples_dir: str = "results/m6/samples"
    m6_n_samples: int = 5
    m6_api_base: str = "http://localhost:8000/v1"
    m6_contradiction_threshold: float = 0.5
    m6_entropy_threshold: float = 1.0
    m6_relevance_threshold: float = 0.25
    folds_path: str = "data/splits/folds.json"
    independent_faithfulness_threshold: float = 0.20
    independent_relevance_threshold: float = 0.10

    independent_v2_model: str = "results/independent_v2/model.joblib"

    limit: int | None = None


BuildCommand = Callable[[CommandContext], list[str]]
Scorer = Callable[["RagSample"], "Prediction"]
ScorerFactory = Callable[[CommandContext], Scorer]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    family: str
    mode: str | None
    build_command: BuildCommand
    demo_runner: str | None
    requires: tuple[str, ...] = field(default_factory=tuple)
    score_keys: tuple[str, ...] = ()
    default_score_expr: str | None = None
    corpus_wide: bool = True
    build_scorer: ScorerFactory | None = None
    # Скрипт, производящий корпус-wide скоры целиком, когда покейсового скорера
    # быть не может: OOF-методы (surface, majority) обязаны видеть весь фолд
    # сразу, поэтому в модель score.py «один кейс -> Prediction» не укладываются.
    corpus_runner: str | None = None


def _maybe_limit(command: list[str], ctx: CommandContext) -> list[str]:
    if ctx.limit is not None:
        command.extend(["--limit", str(ctx.limit)])
    return command


def _dummy(mode: str) -> BuildCommand:
    def build(ctx: CommandContext) -> list[str]:
        return _maybe_limit(
            [
                ctx.python,
                "scripts/run_prompt_baseline.py",
                "--data",
                str(ctx.data),
                "--output",
                str(ctx.predictions_path),
                "--mode",
                mode,
                "--backend",
                "dummy",
                "--dummy-strategy",
                "keyword" if mode == "marker" else "always_reliable",
            ],
            ctx,
        )

    return build


def _prompt(mode: str) -> BuildCommand:
    def build(ctx: CommandContext) -> list[str]:
        return _maybe_limit(
            [
                ctx.python,
                "scripts/run_prompt_baseline.py",
                "--data",
                str(ctx.data),
                "--output",
                str(ctx.predictions_path),
                "--mode",
                mode,
                "--backend",
                "mlx",
                "--model",
                ctx.model,
                "--max-tokens",
                str(ctx.max_tokens),
            ],
            ctx,
        )

    return build


def _lora(mode: str) -> BuildCommand:
    def build(ctx: CommandContext) -> list[str]:
        adapter = ctx.direct_adapter_path if mode == "direct" else ctx.marker_adapter_path
        return [
            ctx.python,
            "scripts/infer.py",
            "--data",
            str(ctx.data),
            "--output",
            str(ctx.predictions_path),
            "--mode",
            mode,
            "--model",
            ctx.model,
            "--adapter-path",
            adapter,
            "--max-tokens",
            str(ctx.max_tokens),
        ]

    return build


def _lettucedetect(ctx: CommandContext) -> list[str]:
    return [
        ctx.python,
        "scripts/infer_lettucedetect.py",
        "--data",
        str(ctx.data),
        "--model",
        ctx.lettucedetect_model,
        "--output",
        str(ctx.predictions_path),
    ]


def _encoder(ctx: CommandContext) -> list[str]:
    checkpoint_dir = ctx.encoder_output_dir or str(ctx.run_dir / "checkpoints")
    return [
        ctx.python,
        "scripts/train_encoder_baseline.py",
        "--data",
        str(ctx.data),
        "--output",
        str(ctx.run_dir / "encoder_binary_metrics.json"),
        "--predictions-output",
        str(ctx.predictions_path),
        "--model",
        ctx.encoder_model,
        "--output-dir",
        checkpoint_dir,
        "--max-length",
        str(ctx.encoder_max_length),
        "--batch-size",
        str(ctx.encoder_batch_size),
        "--epochs",
        str(ctx.encoder_epochs),
        "--learning-rate",
        str(ctx.encoder_learning_rate),
        "--pos-weight-mode",
        ctx.encoder_pos_weight_mode,
    ]


def m3_mode_and_backend(name: str, ctx: CommandContext) -> tuple[str, str]:
    """Разбор имени варианта Метода 3 в (prompt mode, backend).

    Общая точка для build_command и build_scorer: разъехавшись, они дали бы
    subprocess-прогон и score.py на разных промптах при одном имени метода.
    """
    if name in ("m3_openai", "m3_openai_judge"):
        return "zero_shot", "openai" if name == "m3_openai" else "openai_judge"
    return name.removeprefix("m3_"), ctx.m3_backend


def _m3(name: str) -> BuildCommand:
    def build(ctx: CommandContext) -> list[str]:
        m3_mode, backend = m3_mode_and_backend(name, ctx)
        command = [
            ctx.python,
            "scripts/run_m3.py",
            "--data",
            str(ctx.data),
            "--output",
            str(ctx.predictions_path),
            "--mode",
            m3_mode,
            "--backend",
            backend,
            "--model",
            ctx.model,
            "--max-tokens",
            str(ctx.m3_max_tokens),
        ]
        uses_openai = name in ("m3_openai", "m3_openai_judge") or backend in (
            "openai",
            "openai_judge",
        )
        if uses_openai:
            command.extend(
                [
                    "--api-base",
                    ctx.m3_api_base,
                    "--api-key-env",
                    ctx.m3_api_key_env,
                    "--cache-dir",
                    ctx.m3_cache_dir,
                ]
            )
        if name == "m3_openai_judge" or backend == "openai_judge":
            command.extend(["--concurrency", str(ctx.m3_concurrency)])
        if name == "m3_few_shot":
            command.extend(["--examples", ctx.m3_examples])
        elif name == "m3_gepa":
            command.extend(["--prompt-file", ctx.m3_prompt_file])
        if ctx.m3_max_context_chars is not None:
            command.extend(["--max-context-chars", str(ctx.m3_max_context_chars)])
        return _maybe_limit(command, ctx)

    return build


def _m3_perchunk(ctx: CommandContext) -> list[str]:
    """Пофрагментная верификация faithfulness: свой стиль промпта, не режим."""
    command = [
        ctx.python,
        "scripts/run_m3.py",
        "--data",
        str(ctx.data),
        "--output",
        str(ctx.predictions_path),
        "--prompt-style",
        "perchunk",
        "--backend",
        ctx.m3_backend,
        "--model",
        ctx.model,
        "--max-tokens",
        str(ctx.m3_max_tokens),
        "--api-base",
        ctx.m3_api_base,
        "--api-key-env",
        ctx.m3_api_key_env,
        "--cache-dir",
        ctx.m3_cache_dir,
        "--concurrency",
        str(ctx.m3_concurrency),
    ]
    return _maybe_limit(command, ctx)


def _ft_judge(ctx: CommandContext) -> list[str]:
    """Обучение судьи фолд за фолдом: команда собирается на один фолд.

    Реестр не знает номера фолда, поэтому команда указывает нулевой; пять
    фолдов запускаются пятью заданиями DataSphere (``jobs/ft_judge_fold*.yaml``).
    """
    return _maybe_limit(
        [
            ctx.python,
            FT_JUDGE_RUNNER,
            "--data",
            str(ctx.data),
            "--folds",
            ctx.folds_path,
            "--fold",
            "0",
            "--model",
            ctx.model,
            "--predictions-output",
            str(ctx.predictions_path),
            "--output-dir",
            str(ctx.run_dir / "checkpoints"),
        ],
        ctx,
    )


def _m6(ctx: CommandContext) -> list[str]:
    command = [
        ctx.python,
        "scripts/run_m6_pipeline.py",
        "--data",
        str(ctx.data),
        "--samples-dir",
        ctx.m6_samples_dir,
        "--features",
        ctx.m6_features,
        "--output",
        str(ctx.predictions_path),
        "--backend",
        ctx.m6_backend,
        "--n-samples",
        str(ctx.m6_n_samples),
        "--contradiction-threshold",
        str(ctx.m6_contradiction_threshold),
        "--entropy-threshold",
        str(ctx.m6_entropy_threshold),
        "--relevance-threshold",
        str(ctx.m6_relevance_threshold),
    ]
    if ctx.m6_backend == "openai":
        command.extend(["--model", ctx.model, "--api-base", ctx.m6_api_base])
    return _maybe_limit(command, ctx)


def _surface(variant: str) -> BuildCommand:
    def build(ctx: CommandContext) -> list[str]:
        return _maybe_limit(
            [
                ctx.python,
                SURFACE_RUNNER,
                "--variant",
                variant,
                "--data",
                str(ctx.data),
                "--folds",
                ctx.folds_path,
                "--output",
                str(ctx.predictions_path),
            ],
            ctx,
        )

    return build


def _independent(ctx: CommandContext) -> list[str]:
    return _maybe_limit(
        [
            ctx.python,
            "scripts/run_independent.py",
            "--data",
            str(ctx.data),
            "--output",
            str(ctx.predictions_path),
            "--faithfulness-threshold",
            str(ctx.independent_faithfulness_threshold),
            "--relevance-threshold",
            str(ctx.independent_relevance_threshold),
        ],
        ctx,
    )


def _independent_v2(ctx: CommandContext) -> list[str]:
    return _maybe_limit(
        [
            ctx.python,
            "scripts/run_independent_v2.py",
            "--data",
            str(ctx.data),
            "--model",
            ctx.independent_v2_model,
            "--output",
            str(ctx.predictions_path),
        ],
        ctx,
    )


# --------------------------------------------------------------------------- #
# Скореры: один кейс -> Prediction со заполненным scores.
# --------------------------------------------------------------------------- #


def scores_only(prediction: Prediction, scores: dict[str, float]) -> Prediction:
    """Стирает бинарное решение метода, оставляя только скоры.

    Решение принимает протокол оценки (B1) подобранным на train-фолде порогом.
    Если метод оставит своё «жёсткое» решение, в артефакте будут два
    несогласованных ответа — ровно та ошибка, из-за которой бинарные поля
    run_m3.py (порог 0.5) не соответствуют опубликованным числам (0.30/0.63).
    """
    return prediction.model_copy(
        update={
            "faithfulness_pred": 0,
            "relevance_pred": 0,
            "scores": {**prediction.scores, **scores},
        }
    )


def _dummy_scorer(mode: str) -> ScorerFactory:
    strategy = "keyword" if mode == "marker" else "always_reliable"

    def build(ctx: CommandContext) -> Scorer:
        from rag_reliability.dummy_model import DummyPredictor  # noqa: PLC0415
        from rag_reliability.parsing import parse_prediction  # noqa: PLC0415

        predictor = DummyPredictor(strategy=strategy, mode=mode)

        def score(sample: RagSample) -> Prediction:
            raw_output = predictor.predict(sample)
            parsed = parse_prediction(raw_output, sample.id, expect_marker=(mode == "marker"))
            return scores_only(parsed, {})

        return score

    return build


def verdict_scores(prediction: Prediction, prefix: str) -> dict[str, float]:
    """Вероятности вердикта по цепочке logprobs -> regex -> 0.5/0.5.

    У текстовых бэкендов (mlx, openai, dummy) вероятностей нет: разобранный
    вердикт и есть вся информация, а нераспарсенный ответ — незнание, то есть
    0.5, а не «ненадёжно». Кейс не теряется ни в одной ветке.

    Сигнал грубый (три уровня), но это настоящий сигнал метода, а не заглушка:
    любой судья-по-тексту описывается этой же цепочкой, поэтому prompt/lora
    получают её на тех же основаниях, что и Метод 3.
    """
    p_faith = prediction.faithfulness_prob
    p_rel = prediction.relevance_prob
    if p_faith is None or p_rel is None:
        if prediction.invalid_output:
            p_faith, p_rel = 0.5, 0.5
        else:
            p_faith = float(prediction.faithfulness_pred)
            p_rel = float(prediction.relevance_pred)
    return {f"{prefix}.p_faith": float(p_faith), f"{prefix}.p_rel": float(p_rel)}


def _m3_scores(prediction: Prediction) -> dict[str, float]:
    return verdict_scores(prediction, "m3")


def _prompt_scorer(mode: str, family: str) -> ScorerFactory:
    """Судья по сгенерированному тексту: zero-shot промпт (family=prompt) или LoRA."""

    def build(ctx: CommandContext) -> Scorer:
        from rag_reliability.mlx_backend import make_generate_fn  # noqa: PLC0415
        from rag_reliability.parsing import parse_prediction  # noqa: PLC0415
        from rag_reliability.prompts import (  # noqa: PLC0415
            build_direct_prompt,
            build_marker_prompt,
        )

        adapter_path = None
        if family == "lora":
            adapter_path = (
                ctx.direct_adapter_path if mode == "direct" else ctx.marker_adapter_path
            )
        build_prompt = build_direct_prompt if mode == "direct" else build_marker_prompt
        generate_fn = make_generate_fn(ctx.model, ctx.max_tokens, adapter_path=adapter_path)

        def score(sample: RagSample) -> Prediction:
            raw_output = generate_fn(build_prompt(sample))
            parsed = parse_prediction(raw_output, sample.id, expect_marker=(mode == "marker"))
            return scores_only(parsed, verdict_scores(parsed, family))

        return score

    return build


def _m3_scorer(name: str) -> ScorerFactory:  # noqa: C901 - одна ветка на бэкенд
    def build(ctx: CommandContext) -> Scorer:  # noqa: C901
        from rag_reliability.methods.m3 import (  # noqa: PLC0415
            build_system_prompt,
            build_user_prompt,
            parse_m3_prediction,
        )
        from rag_reliability.schema import Prediction as PredictionModel  # noqa: PLC0415

        mode, backend = m3_mode_and_backend(name, ctx)
        system_prompt = build_system_prompt(
            mode,
            examples_path=ctx.m3_examples if name == "m3_few_shot" else None,
            prompt_file=ctx.m3_prompt_file if name == "m3_gepa" else None,
        )

        if backend == "openai_judge":
            from rag_reliability.methods.m3.judge_client import JudgeClient  # noqa: PLC0415

            client = JudgeClient(
                model=ctx.model,
                api_base=ctx.m3_api_base,
                api_key=os.environ.get(ctx.m3_api_key_env, ""),
                cache_dir=ctx.m3_cache_dir,
            )

            def score_judge(sample: RagSample) -> Prediction:
                user_prompt = build_user_prompt(sample, ctx.m3_max_context_chars)
                p_faith, p_rel, meta = client.judge(
                    system_prompt, user_prompt, max_tokens=ctx.m3_max_tokens
                )
                return PredictionModel(
                    id=sample.id,
                    faithfulness_pred=0,
                    relevance_pred=0,
                    raw_output=str(meta["raw"]),
                    invalid_output=meta["method"] == "default",
                    faithfulness_prob=p_faith,
                    relevance_prob=p_rel,
                    prob_method=str(meta["method"]),
                    scores={"m3.p_faith": float(p_faith), "m3.p_rel": float(p_rel)},
                )

            return score_judge

        if backend == "dummy":
            from rag_reliability.dummy_model import DummyPredictor  # noqa: PLC0415
            from rag_reliability.parsing import parse_prediction  # noqa: PLC0415

            predictor = DummyPredictor(strategy=ctx.m3_dummy_strategy, mode="direct")

            def score_dummy(sample: RagSample) -> Prediction:
                parsed = parse_prediction(predictor.predict(sample), sample.id)
                return scores_only(parsed, _m3_scores(parsed))

            return score_dummy

        if backend == "openai":
            from rag_reliability.methods.m3.openai_client import (  # noqa: PLC0415
                CachedChatClient,
            )

            chat_client = CachedChatClient(
                model=ctx.model,
                api_base=ctx.m3_api_base,
                api_key=os.environ.get(ctx.m3_api_key_env, ""),
                cache_dir=ctx.m3_cache_dir,
            )

            def score_openai(sample: RagSample) -> Prediction:
                raw_output = chat_client.chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": build_user_prompt(sample, ctx.m3_max_context_chars),
                        },
                    ],
                    max_tokens=ctx.m3_max_tokens,
                )
                parsed = parse_m3_prediction(raw_output, sample.id)
                return scores_only(parsed, _m3_scores(parsed))

            return score_openai

        from rag_reliability.mlx_backend import make_generate_fn  # noqa: PLC0415

        generate_fn = make_generate_fn(ctx.model, ctx.m3_max_tokens)

        def score_mlx(sample: RagSample) -> Prediction:
            user_prompt = build_user_prompt(sample, ctx.m3_max_context_chars)
            raw_output = generate_fn(f"{system_prompt}\n\n{user_prompt}")
            parsed = parse_m3_prediction(raw_output, sample.id)
            return scores_only(parsed, _m3_scores(parsed))

        return score_mlx

    return build


def build_m3_chat_client(ctx: CommandContext) -> Any:
    """Клиент судьи под пофрагментный путь: асинхронный при concurrency > 1.

    Пофрагментная верификация шлёт по запросу на чанк, то есть 5-15 запросов на
    кейс. Синхронный клиент делает их по очереди, и корпусный прогон
    растягивается на часы там, где узкое место — сеть, а не GPU.
    """
    from rag_reliability.methods.m3.judge_client import (  # noqa: PLC0415
        AsyncJudgeClient,
        JudgeClient,
    )

    common = {
        "model": ctx.model,
        "api_base": ctx.m3_api_base,
        "api_key": os.environ.get(ctx.m3_api_key_env, ""),
        "cache_dir": ctx.m3_cache_dir,
    }
    if ctx.m3_concurrency > 1:
        return AsyncJudgeClient(concurrency=ctx.m3_concurrency, **common)
    return JudgeClient(**common)


def _m3_perchunk_scorer(ctx: CommandContext) -> Scorer:
    """Пофрагментные фичи faithfulness одного кейса.

    Relevance по контракту C3 чанки не получает: её определение опирается
    только на вопрос и ответ. Поэтому ``m3.p_rel`` здесь не появляется, а
    выражение скора по умолчанию строится на ``m3.max_chunk_score``.
    """
    from rag_reliability.methods.m3.perchunk import score_per_chunk  # noqa: PLC0415
    from rag_reliability.schema import Prediction as PredictionModel  # noqa: PLC0415

    client = build_m3_chat_client(ctx)

    def score(sample: RagSample) -> Prediction:
        features = score_per_chunk(sample, client)
        return PredictionModel(
            id=sample.id,
            faithfulness_pred=0,
            relevance_pred=0,
            prob_method="perchunk_logprobs",
            scores={key: float(value) for key, value in features.items()},
        )

    return score


def _lettucedetect_scorer(ctx: CommandContext) -> Scorer:
    import joblib  # noqa: PLC0415

    from rag_reliability.methods.lettucedetect.features import (  # noqa: PLC0415
        FeatureConfig,
        extract_features,
        make_detector,
    )
    from rag_reliability.schema import Prediction as PredictionModel  # noqa: PLC0415

    artifact = joblib.load(ctx.lettucedetect_model)
    saved_config = artifact["feature_config"]
    config = FeatureConfig(
        model_path=saved_config["model_path"],
        threshold=saved_config["threshold"],
        device=saved_config["device"],
    )
    detector = make_detector(config)

    def score(sample: RagSample) -> Prediction:
        row = extract_features([sample], detector, config.threshold, desc="lettucedetect")[0]
        return PredictionModel(
            id=sample.id,
            faithfulness_pred=0,
            relevance_pred=0,
            scores={
                "ld.max_unsup": float(row[0]),
                "ld.mean_unsup": float(row[1]),
                "ld.frac_unsup": float(row[2]),
            },
        )

    return score


def _independent_scorer(ctx: CommandContext) -> Scorer:
    from rag_reliability.methods.independent.predict import predict_independent  # noqa: PLC0415

    def score(sample: RagSample) -> Prediction:
        prediction = predict_independent(
            sample,
            faithfulness_threshold=ctx.independent_faithfulness_threshold,
            relevance_threshold=ctx.independent_relevance_threshold,
        )
        if prediction.raw_output is None:
            raise ValueError(
                f"independent evaluator returned no diagnostics for sample {sample.id!r}; "
                "cannot derive ind.* scores"
            )
        diagnostics = json.loads(prediction.raw_output)
        for key in ("faithfulness_score", "relevance_score"):
            if key not in diagnostics:
                raise KeyError(
                    f"independent diagnostics for sample {sample.id!r} lack {key!r}; "
                    f"present keys: {sorted(diagnostics)[:8]}"
                )
        # Единственный метод, который по природе бинарен: _pred остаются осмысленными.
        return prediction.model_copy(
            update={
                "scores": {
                    "ind.faith_score": float(diagnostics["faithfulness_score"]),
                    "ind.rel_score": float(diagnostics["relevance_score"]),
                }
            }
        )

    return score


SURFACE_RUNNER = "scripts/run_surface_baseline.py"
_SURF_FEATURE_KEYS = tuple(f"surf.{key}" for key in FEATURE_KEYS)
_SURF_SCORE_KEYS = (*_SURF_FEATURE_KEYS, "surf.p_faith", "surf.p_rel")
_SURF_SCORE_EXPR = "surf.p_faith * surf.p_rel"

_M3_SCORE_KEYS = ("m3.p_faith", "m3.p_rel")
_M3_SCORE_EXPR = "m3.p_faith * m3.p_rel"

FT_JUDGE_RUNNER = "scripts/train_ft_judge.py"
#: Пофрагментный путь даёт фичи, а не вероятность оси: агрегат выбирает стэкер.
_PERCHUNK_SCORE_KEYS = (
    "m3.max_chunk_score",
    "m3.mean_chunk_score",
    "m3.chunk_disagreement",
    "m3.n_supporting",
    "m3.argmax_chunk",
)
_PERCHUNK_SCORE_EXPR = "m3.max_chunk_score"
_PROMPT_SCORE_KEYS = ("prompt.p_faith", "prompt.p_rel")
_PROMPT_SCORE_EXPR = "prompt.p_faith * prompt.p_rel"
_LORA_SCORE_KEYS = ("lora.p_faith", "lora.p_rel")
_LORA_SCORE_EXPR = "lora.p_faith * lora.p_rel"


METHODS: dict[str, MethodSpec] = {
    "dummy_direct": MethodSpec(
        "dummy_direct", "Dummy — direct", "dummy", "direct", _dummy("direct"), "dummy",
        build_scorer=_dummy_scorer("direct"),
    ),
    "dummy_marker": MethodSpec(
        "dummy_marker", "Dummy — marker", "dummy", "marker", _dummy("marker"), "dummy",
        build_scorer=_dummy_scorer("marker"),
    ),
    "prompt_direct": MethodSpec(
        "prompt_direct", "Zero-shot prompt — direct", "prompt", "direct", _prompt("direct"),
        "prompt", ("MLX model",),
        score_keys=_PROMPT_SCORE_KEYS, default_score_expr=_PROMPT_SCORE_EXPR,
        build_scorer=_prompt_scorer("direct", "prompt"),
    ),
    "prompt_marker": MethodSpec(
        "prompt_marker", "Zero-shot prompt — marker", "prompt", "marker", _prompt("marker"),
        "prompt", ("MLX model",),
        score_keys=_PROMPT_SCORE_KEYS, default_score_expr=_PROMPT_SCORE_EXPR,
        build_scorer=_prompt_scorer("marker", "prompt"),
    ),
    "lora_direct": MethodSpec(
        "lora_direct", "LoRA — direct", "lora", "direct", _lora("direct"), "lora",
        ("results/adapters_direct",),
        score_keys=_LORA_SCORE_KEYS, default_score_expr=_LORA_SCORE_EXPR,
        build_scorer=_prompt_scorer("direct", "lora"),
    ),
    "lora_marker": MethodSpec(
        "lora_marker", "LoRA — marker", "lora", "marker", _lora("marker"), "lora",
        ("results/adapters_marker",),
        score_keys=_LORA_SCORE_KEYS, default_score_expr=_LORA_SCORE_EXPR,
        build_scorer=_prompt_scorer("marker", "lora"),
    ),
    "lettucedetect": MethodSpec(
        "lettucedetect", "LettuceDetect features", "lettucedetect", None, _lettucedetect,
        "lettucedetect", ("results/lettucedetect/classifier.joblib",),
        score_keys=("ld.max_unsup", "ld.mean_unsup", "ld.frac_unsup"),
        default_score_expr="1 - ld.max_unsup",
        build_scorer=_lettucedetect_scorer,
    ),
    "encoder": MethodSpec(
        "encoder", "RuModernBERT encoder", "encoder", None, _encoder, "encoder",
        ("results/encoder_checkpoints_512_best",),
        score_keys=("enc.logit",),
        default_score_expr="enc.logit",
        corpus_wide=False,
    ),
    "m3_zero_shot": MethodSpec(
        "m3_zero_shot", "Method 3 — zero-shot judge", "m3", None, _m3("m3_zero_shot"), "m3",
        ("MLX model",),
        score_keys=_M3_SCORE_KEYS, default_score_expr=_M3_SCORE_EXPR,
        build_scorer=_m3_scorer("m3_zero_shot"),
    ),
    "m3_few_shot": MethodSpec(
        "m3_few_shot", "Method 3 — few-shot judge", "m3", None, _m3("m3_few_shot"), "m3",
        ("configs/few_shot.yaml",),
        score_keys=_M3_SCORE_KEYS, default_score_expr=_M3_SCORE_EXPR,
        build_scorer=_m3_scorer("m3_few_shot"),
    ),
    "m3_gepa": MethodSpec(
        "m3_gepa", "Method 3 — GEPA prompt", "m3", None, _m3("m3_gepa"), None,
        ("configs/m3_gepa_prompt.txt",),
        score_keys=_M3_SCORE_KEYS, default_score_expr=_M3_SCORE_EXPR,
        build_scorer=_m3_scorer("m3_gepa"),
    ),
    "m3_openai": MethodSpec(
        "m3_openai", "Method 3 — OpenAI endpoint", "m3", None, _m3("m3_openai"), None,
        ("OpenAI-compatible endpoint",),
        score_keys=_M3_SCORE_KEYS, default_score_expr=_M3_SCORE_EXPR,
        build_scorer=_m3_scorer("m3_openai"),
    ),
    "m3_openai_judge": MethodSpec(
        "m3_openai_judge", "Method 3 — OpenAI logprob judge", "m3", None,
        _m3("m3_openai_judge"), None, ("OpenAI-compatible endpoint",),
        score_keys=_M3_SCORE_KEYS, default_score_expr=_M3_SCORE_EXPR,
        build_scorer=_m3_scorer("m3_openai_judge"),
    ),
    "m3_perchunk": MethodSpec(
        "m3_perchunk", "Method 3 — per-chunk faithfulness", "m3", None, _m3_perchunk, None,
        ("OpenAI-compatible endpoint",),
        score_keys=_PERCHUNK_SCORE_KEYS, default_score_expr=_PERCHUNK_SCORE_EXPR,
        build_scorer=_m3_perchunk_scorer,
    ),
    "ft_judge": MethodSpec(
        "ft_judge", "Method 3 — fine-tuned judge (per fold)", "m3", None, _ft_judge, None,
        ("data/splits/folds.json", "GPU >= 70 GB"),
        score_keys=_M3_SCORE_KEYS, default_score_expr=_M3_SCORE_EXPR,
        # Один прогон = один фолд, поэтому и не corpus_wide, и не corpus_runner:
        # корпус покрывают пять запусков, а не один. Куда идти — в NOT_CASE_WISE.
        corpus_wide=False,
    ),
    "m6_selfcheck": MethodSpec(
        "m6_selfcheck",
        "Method 6 — SelfCheck features",
        "m6",
        None,
        _m6,
        None,
        ("m6 pipeline (dummy offline; real: --m6-backend openai + .[m6])",),
        score_keys=("m6.contra_mean", "m6.entropy", "m6.cos_q_a"),
        default_score_expr="(1 - m6.contra_mean) * m6.cos_q_a",
        corpus_wide=False,
    ),
    "surface": MethodSpec(
        "surface", "Surface features + OOF logreg", "surface", None, _surface("surface"),
        None, ("data/splits/folds.json",),
        score_keys=_SURF_SCORE_KEYS, default_score_expr=_SURF_SCORE_EXPR,
        corpus_runner=SURFACE_RUNNER,
    ),
    "majority": MethodSpec(
        "majority", "Majority base rate (OOF)", "surface", None, _surface("majority"),
        None, ("data/splits/folds.json",),
        score_keys=("surf.p_faith", "surf.p_rel"),
        corpus_runner=SURFACE_RUNNER,
    ),
    "independent": MethodSpec(
        "independent", "Independent rule-based evaluator", "independent", None, _independent,
        "independent",
        score_keys=("ind.faith_score", "ind.rel_score"),
        default_score_expr="ind.faith_score * ind.rel_score",
        build_scorer=_independent_scorer,
    ),
    "independent_v2": MethodSpec(
        "independent_v2", "Independent evaluator V2 — learned features", "independent", None,
        _independent_v2, None,
        ("results/independent_v2/model.joblib",),
        score_keys=("ind.p_faith", "ind.p_rel"),
        default_score_expr="ind.p_faith * ind.p_rel",
        # Batch-only until a per-case scorer is wired; train/infer via scripts/*.
        corpus_wide=False,
    ),
}


def all_method_names() -> tuple[str, ...]:
    return tuple(METHODS)


def get(name: str) -> MethodSpec:
    try:
        return METHODS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown method {name!r}; available: {', '.join(METHODS)}") from exc


def resolve_names(raw: str) -> list[str]:
    if raw.strip() == "all":
        return list(all_method_names())
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Available: {list(METHODS)}")
    return names


# Кто в волне 3 доводит метод до корпус-wide скоринга (для сообщения об ошибке и PR).
WAVE3_OWNER: dict[str, str] = {
    "encoder": "out-of-fold scoring is task C2 (task/C2-encoder)",
    "m6_selfcheck": "grounding rewrite is task C4 (task/C4-m6-grounding)",
    # Не «ещё не сделано», а «по построению не покейсово»: обучение видит фолд
    # целиком, и корпус покрывают пять запусков, а не один.
    "ft_judge": f"training runs fold by fold: {FT_JUDGE_RUNNER} --fold N",
}


def contract_version(spec: MethodSpec) -> str:
    """Версия метода для run.yaml: хэш его объявленного контракта.

    Меняется ровно тогда, когда меняются имя, семейство, режим, набор ключей,
    выражение по умолчанию или corpus_wide, — то есть когда старый артефакт
    перестаёт быть сопоставимым с новым. Версия реализации пришпилена git-хэшем
    в том же run.yaml; вручную поддерживаемый номер здесь протух бы первым же
    рефакторингом.
    """
    payload = json.dumps(
        {
            "name": spec.name,
            "family": spec.family,
            "mode": spec.mode,
            "score_keys": list(spec.score_keys),
            "default_score_expr": spec.default_score_expr,
            "corpus_wide": spec.corpus_wide,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return "contract-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_scorer(name: str, ctx: CommandContext) -> Scorer:
    """Готовый per-sample скорер метода или понятный отказ.

    Отказ содержит имя задачи волны 3, которая обязана научить метод скорить
    весь корпус, — иначе `score.py` падает загадочным None.
    """
    spec = get(name)
    if spec.build_scorer is None:
        if spec.corpus_runner is not None:
            raise ValueError(
                f"Method {name!r} scores out-of-fold and cannot be run case by case; "
                f"use {spec.corpus_runner}"
            )
        reason = WAVE3_OWNER.get(name, "see docs/handoff/HANDOFF.md §5")
        raise ValueError(
            f"Method {name!r} has no corpus-wide scorer "
            f"(corpus_wide={spec.corpus_wide}); {reason}"
        )
    return spec.build_scorer(ctx)


class ScoresValidationError(ValueError):
    """Артефакт scores.jsonl не соответствует контракту метода."""


def validate_scores_file(
    path: str | Path,
    spec: MethodSpec,
    expected_n: int | None = None,
) -> None:
    """Проверяет: нет дублей id, все score_keys присутствуют в каждой строке,
    значения конечны, при expected_n — совпадает количество.

    Отсутствие ключа — ошибка. Никаких .get(key, default): в M6 молчаливый
    дефолт привёл к тому, что битая строка признаков трактовалась как
    идеально надёжный кейс.
    """
    path = Path(path)
    if not path.exists():
        raise ScoresValidationError(f"Scores file not found: {path}")

    seen: set[str] = set()
    n_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoresValidationError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict) or "id" not in row:
                raise ScoresValidationError(f"Row at {path}:{line_number} has no 'id'")
            sample_id = str(row["id"])
            if sample_id in seen:
                raise ScoresValidationError(
                    f"Duplicate prediction id {sample_id!r} at {path}:{line_number}"
                )
            seen.add(sample_id)
            n_rows += 1

            scores = row.get("scores")
            if scores is None:
                if spec.score_keys:
                    raise ScoresValidationError(
                        f"Row {sample_id!r} at {path}:{line_number} has no 'scores'; "
                        f"method {spec.name!r} declares {list(spec.score_keys)}"
                    )
                scores = {}
            if not isinstance(scores, dict):
                raise ScoresValidationError(
                    f"Field 'scores' for sample {sample_id!r} at {path}:{line_number} "
                    f"must be an object, got {type(scores).__name__}"
                )

            for key in spec.score_keys:
                if key not in scores:
                    raise ScoresValidationError(
                        f"Missing score key {key!r} for sample {sample_id!r} at "
                        f"{path}:{line_number}; present: {sorted(scores)[:8]}"
                    )

            # Проверяются ВСЕ ключи, а не только объявленные: незаявленный NaN
            # тоже попадёт в стэкер и тоже всё сломает — контракт задаёт минимум
            # содержимого, а не разрешение писать мусор мимо него.
            for key, value in scores.items():
                if not isinstance(value, int | float) or isinstance(value, bool):
                    raise ScoresValidationError(
                        f"Score {key!r} for sample {sample_id!r} at {path}:{line_number} "
                        f"must be a number, got {type(value).__name__}"
                    )
                if not math.isfinite(float(value)):
                    raise ScoresValidationError(
                        f"Score {key!r} for sample {sample_id!r} at {path}:{line_number} "
                        f"is not finite: {value!r}"
                    )

    if expected_n is not None and n_rows != expected_n:
        raise ScoresValidationError(
            f"{path} has {n_rows} row(s), expected {expected_n}"
        )
