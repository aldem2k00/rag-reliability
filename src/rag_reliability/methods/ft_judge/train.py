"""Обучение судьи на одном фолде ``folds.json`` с контролем схлопывания.

Три решения, ради которых модуль появился.

**Разбиение не создаётся здесь.** Прежний ноутбук звал ``split_samples`` —
собственный, не group-aware протокол. Фолд читается из ``folds.json``: кейс
скорится моделью, которая его не видела, иначе стэкер C1 получит признак,
видевший свою метку.

**Схлопывание ловится после каждой эпохи, а не постфактум.** Прогон на 1.5B
схлопнулся в константный вердикт (1,1) при дисбалансе 72/28 и прошёл
незамеченным. Здесь после каждой эпохи считается ``degenerate_rate`` по
held-out части фолда, и прогон с ``const_share > 0.98`` помечается
``collapsed`` — такая конфигурация не имеет права считаться лучшей.

**Вероятность приходит из логпробов вердикта.** Ни на эпохе, ни в артефакте
вердикт не парсится из сгенерированного текста: якорь оси форсируется, и
сравниваются логиты токенов PASS/FAIL на позиции вердикта. Это и дешевле
генерации, и совпадает с тем, как читает вероятность инференс C3.

Один вызов обучает один фолд: пять фолдов — пять заданий DataSphere по ~1.5 ч,
и обрыв одного не стоит остальных.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_reliability.methods.ft_judge.data import (
    MODES,
    JudgeExample,
    build_examples,
    compute_pos_weight,
    oversample_negatives,
)
from rag_reliability.methods.ft_judge.predict import AxisProbs, decisions_from_probs
from rag_reliability.metrics import degenerate_rate

if TYPE_CHECKING:
    from rag_reliability.methods.surface.oof import Folds
    from rag_reliability.schema import RagSample

logger = logging.getLogger(__name__)

POS_WEIGHT_MODES: tuple[str, ...] = ("none", "balanced")
TUNINGS: tuple[str, ...] = ("lora", "full")
SAVE_STRATEGIES: tuple[str, ...] = ("no", "epoch")

#: 7B в bf16 с LoRA r=256 и контекстом 2048 не помещается в меньшее.
MIN_VRAM_GB = 70


@dataclass(frozen=True)
class FtConfig:
    """Гиперпараметры прогона. Всё, что попадает в ``run.yaml``."""

    model: str = "Qwen/Qwen2.5-7B-Instruct"
    mode: str = "direct"
    tuning: str = "lora"
    lora_r: int = 256
    lora_alpha: int = 512
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    epochs: float = 3.0
    max_length: int = 2048
    batch_size: int = 1
    grad_accum: int = 8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    pos_weight_mode: str = "balanced"
    oversample_negatives: bool = True
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    seed: int = 42
    output_dir: str = "results/ft_judge"
    prompts_dir: str | None = None
    #: Ассерт VRAM снимается только осознанно — смоук на маленькой карте.
    allow_small_gpu: bool = False
    push_to_hub: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.tuning not in TUNINGS:
            raise ValueError(f"tuning must be one of {TUNINGS}, got {self.tuning!r}")
        if self.pos_weight_mode not in POS_WEIGHT_MODES:
            raise ValueError(
                f"pos_weight_mode must be one of {POS_WEIGHT_MODES}, "
                f"got {self.pos_weight_mode!r}"
            )
        if self.save_strategy not in SAVE_STRATEGIES:
            raise ValueError(
                f"save_strategy must be one of {SAVE_STRATEGIES}, got {self.save_strategy!r}"
            )
        for name in ("lora_r", "lora_alpha", "max_length", "batch_size", "grad_accum"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.save_total_limit < 1:
            raise ValueError(f"save_total_limit must be >= 1, got {self.save_total_limit}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")

    @property
    def n_epochs(self) -> int:
        """Число эпох, после каждой из которых обязана быть диагностика."""
        return math.ceil(self.epochs)


@dataclass(frozen=True)
class EpochLog:
    """Диагностика одной эпохи. Форма совпадает с ``encoder_diagnostics.json``."""

    repeat: int
    fold: int
    epoch: int
    n_held_out: int
    const_share: float
    output_entropy: float
    is_degenerate: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "repeat": self.repeat,
            "fold": self.fold,
            "epoch": self.epoch,
            "n_held_out": self.n_held_out,
            "const_share": self.const_share,
            "output_entropy": self.output_entropy,
            "is_degenerate": self.is_degenerate,
        }


EpochHook = Callable[[int, Sequence[AxisProbs]], EpochLog]


@dataclass(frozen=True)
class FoldRequest:
    """Всё, что нужно обучению одного фолда, включая обязательный хук эпохи."""

    repeat: int
    fold: int
    train_examples: tuple[JudgeExample, ...]
    held_out: tuple[RagSample, ...]
    config: FtConfig
    on_epoch_end: EpochHook
    pos_weight: float = 1.0
    resume: bool = False


@dataclass(frozen=True)
class FoldOutcome:
    """Результат обучения фолда: вероятности held-out и путь к чекпоинту."""

    probs: tuple[AxisProbs, ...]
    checkpoint: str | None = None


FoldTrainer = Callable[[FoldRequest], FoldOutcome]


@dataclass
class FoldResult:
    """Вероятности фолда плюс всё, без чего прогон нельзя честно описать."""

    probs: dict[str, AxisProbs]
    epochs: list[EpochLog] = field(default_factory=list)
    checkpoint: str | None = None
    repeat: int = 0
    fold: int = 0
    n_train: int = 0
    pos_weight: float = 1.0
    class_balance: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def collapsed(self) -> bool:
        return self.collapse_reason is not None

    @property
    def collapse_reason(self) -> str | None:
        """Почему прогон считается схлопнувшимся — или ``None``.

        Два независимых повода. ``pooled`` — вердикты на всём held-out
        практически одинаковы. ``last_epoch`` — та же проверка после последней
        эпохи: она ловит случай, когда модель схлопнулась именно к концу
        обучения, а усреднение по эпохам это смазало бы.
        """
        if not self.probs:
            return None
        if bool(degenerate_rate(decisions_from_probs(self.probs))["is_degenerate"]):
            return "pooled"
        last = max(self.epochs, key=lambda log: log.epoch, default=None)
        if last is not None and last.is_degenerate:
            return "last_epoch"
        return None

    def diagnostics(self) -> dict[str, Any]:
        summary = degenerate_rate(decisions_from_probs(self.probs))
        return {
            "n_scored": len(self.probs),
            "n_train": self.n_train,
            "repeat": self.repeat,
            "fold": self.fold,
            "pos_weight": self.pos_weight,
            "class_balance": self.class_balance,
            "collapsed": self.collapsed,
            "collapse_reason": self.collapse_reason,
            "const_share": float(summary["const_share"]),
            "output_entropy": float(summary["output_entropy"]),
            "checkpoint": self.checkpoint,
            "epochs": [log.as_dict() for log in self.epochs],
        }


def make_epoch_hook(*, repeat: int, fold: int, sink: list[EpochLog]) -> EpochHook:
    """Хук, который считает ``degenerate_rate`` и пишет его в лог и в ``sink``."""

    def on_epoch_end(epoch: int, probs: Sequence[AxisProbs]) -> EpochLog:
        if not probs:
            raise ValueError(
                f"repeat {repeat} fold {fold} epoch {epoch}: no held-out probabilities to diagnose"
            )
        summary = degenerate_rate(
            decisions_from_probs({str(index): item for index, item in enumerate(probs)})
        )
        log = EpochLog(
            repeat=repeat,
            fold=fold,
            epoch=epoch,
            n_held_out=len(probs),
            const_share=float(summary["const_share"]),
            output_entropy=float(summary["output_entropy"]),
            is_degenerate=bool(summary["is_degenerate"]),
        )
        logger.info(
            "repeat %d fold %d epoch %d: const_share=%.4f entropy=%.4f degenerate=%s",
            repeat,
            fold,
            epoch,
            log.const_share,
            log.output_entropy,
            log.is_degenerate,
        )
        sink.append(log)
        return log

    return on_epoch_end


def fold_partition(
    samples: Sequence[RagSample], folds: Folds, *, repeat: int, fold: int
) -> tuple[tuple[RagSample, ...], tuple[RagSample, ...]]:
    """(train, held-out) одного фолда. Кейсы без фолда не участвуют ни там, ни там.

    Кейсы вне ``folds.json`` — одна oversized-группа; предсказать их
    out-of-fold нечем, а класть их в train одного фолда значило бы отдать их
    же соседнему фолду в held-out.
    """
    if not 0 <= repeat < folds.n_repeats:
        raise ValueError(
            f"repeat must be in [0, {folds.n_repeats}), got {repeat}; "
            "folds.json declares fewer repeats than requested"
        )
    if not 0 <= fold < folds.n_folds:
        raise ValueError(f"fold must be in [0, {folds.n_folds}), got {fold}")
    train: list[RagSample] = []
    held_out: list[RagSample] = []
    for sample in samples:
        assignment = folds.assignment.get(sample.id)
        if assignment is None:
            continue
        (held_out if assignment[repeat] == fold else train).append(sample)
    return tuple(train), tuple(held_out)


def train_one_fold(
    samples: Sequence[RagSample],
    folds: Folds,
    config: FtConfig,
    *,
    fold: int,
    repeat: int = 0,
    resume: bool = False,
    train_fold: FoldTrainer | None = None,
) -> FoldResult:
    """Обучить фолд и вернуть вероятности его held-out части.

    ``train_fold`` вынесен в параметр не ради гибкости, а ради тестируемости:
    раскладка по фолдам, взвешивание, oversampling и контроль схлопывания
    обязаны проверяться без GPU.
    """
    if not samples:
        raise ValueError("Cannot train on an empty corpus")
    train_samples, held_out = fold_partition(samples, folds, repeat=repeat, fold=fold)
    if not held_out:
        raise ValueError(
            f"repeat {repeat} fold {fold} has no held-out case in this run; "
            "wrong fold number, wrong corpus or too small a --limit"
        )
    if not train_samples:
        raise ValueError(f"repeat {repeat} fold {fold} leaves no training part")

    examples = build_examples(train_samples, mode=config.mode, prompts_dir=config.prompts_dir)
    from rag_reliability.methods.ft_judge.data import class_balance  # noqa: PLC0415

    balance = class_balance(examples)
    pos_weight = compute_pos_weight(
        [example.label for example in examples], config.pos_weight_mode
    )
    if config.oversample_negatives:
        examples = oversample_negatives(examples, seed=config.seed)

    result = FoldResult(
        probs={},
        repeat=repeat,
        fold=fold,
        n_train=len(examples),
        pos_weight=pos_weight,
        class_balance=balance,
    )
    trainer = train_fold if train_fold is not None else train_fold_transformers
    before = len(result.epochs)
    outcome = trainer(
        FoldRequest(
            repeat=repeat,
            fold=fold,
            train_examples=tuple(examples),
            held_out=held_out,
            config=config,
            on_epoch_end=make_epoch_hook(repeat=repeat, fold=fold, sink=result.epochs),
            pos_weight=pos_weight,
            resume=resume,
        )
    )
    if len(outcome.probs) != len(held_out):
        raise ValueError(
            f"repeat {repeat} fold {fold}: trainer returned {len(outcome.probs)} probability "
            f"pair(s) for {len(held_out)} held-out case(s)"
        )
    logged = len(result.epochs) - before
    # Возобновлённый прогон логирует только доученные эпохи — это не пропуск
    # диагностики, а её отсутствие там, где обучения в этом процессе не было.
    if not resume and logged != config.n_epochs:
        raise ValueError(
            f"repeat {repeat} fold {fold}: collapse diagnostics ran {logged} time(s) for "
            f"{config.n_epochs} epoch(s); degenerate_rate is mandatory after every epoch"
        )
    for sample, probs in zip(held_out, outcome.probs, strict=True):
        result.probs[sample.id] = probs
    result.checkpoint = outcome.checkpoint

    if result.collapsed:
        logger.warning(
            "fold %d collapsed (%s): const_share=%.4f — configuration is excluded from "
            "best-config choice",
            fold,
            result.collapse_reason,
            float(degenerate_rate(decisions_from_probs(result.probs))["const_share"]),
        )
    return result


# --------------------------------------------------------------------------- #
# Реализация обучения фолда на transformers/peft. Всё тяжёлое — только здесь.
# --------------------------------------------------------------------------- #


def set_seed(seed: int) -> None:
    import random  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_vram(config: FtConfig) -> float:
    """Ассерт VRAM до загрузки весов: OOM на третьем часу дороже отказа на первой минуте."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        if config.allow_small_gpu:
            logger.warning("CUDA is unavailable; running on CPU because allow_small_gpu is set")
            return 0.0
        raise RuntimeError(
            "Fine-tuning the judge needs a CUDA device; none is visible. "
            "Run on a GPU configuration (g2.1) or pass --allow-small-gpu for a smoke run."
        )
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb < MIN_VRAM_GB and not config.allow_small_gpu:
        raise RuntimeError(
            f"Visible GPU has {total_gb:.1f} GB of VRAM, less than the {MIN_VRAM_GB} GB this "
            f"configuration needs ({config.model}, LoRA r={config.lora_r}, "
            f"max_length={config.max_length}). Use a bigger configuration or --allow-small-gpu."
        )
    logger.info("GPU: %s, %.1f GB VRAM", torch.cuda.get_device_name(0), total_gb)
    return total_gb


def _prompt_text(tokenizer: Any, example: JudgeExample) -> str:
    """Промпт в чат-шаблоне модели — ровно тот же вызов, что на инференсе."""
    return tokenizer.apply_chat_template(
        example.messages, tokenize=False, add_generation_prompt=True
    )


def encode_example(
    tokenizer: Any, example: JudgeExample, max_length: int
) -> dict[str, list[int]]:
    """Токены промпта и завершения; лосс считается только по завершению.

    Промпт маскируется ``-100``: учить модель воспроизводить собственную
    инструкцию незачем, а на длинном контексте это ещё и заглушило бы градиент
    от единственных двух токенов, которые нас интересуют.

    Усечение — слева по промпту: вердикт и якорь обязаны остаться в окне.
    """
    prompt_ids = tokenizer(_prompt_text(tokenizer, example), add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(example.completion, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        completion_ids = [*completion_ids, tokenizer.eos_token_id]
    budget = max_length - len(completion_ids)
    if budget < 1:
        raise ValueError(
            f"max_length={max_length} leaves no room for the completion of sample "
            f"{example.sample_id!r} ({len(completion_ids)} token(s))"
        )
    prompt_ids = prompt_ids[-budget:]
    return {
        "input_ids": [*prompt_ids, *completion_ids],
        "labels": [-100] * len(prompt_ids) + list(completion_ids),
    }


def _collate(batch: Sequence[dict[str, list[int]]], pad_id: int) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    width = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[row, :length] = torch.tensor(item["labels"], dtype=torch.long)
        attention_mask[row, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def verdict_token_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Варианты первого токена PASS и FAIL: с ведущим пробелом и без него.

    Токенизаторы BPE кодируют « PASS» и «PASS» разными идентификаторами, и
    какой из них окажется первым после «FAITHFULNESS:», зависит от шаблона.
    Берём оба и суммируем вероятности — иначе половина массы вердикта молча
    теряется.
    """
    pass_ids: list[int] = []
    fail_ids: list[int] = []
    for text, sink in ((" PASS", pass_ids), ("PASS", pass_ids), (" FAIL", fail_ids), ("FAIL", fail_ids)):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            sink.append(ids[0])
    if not pass_ids or not fail_ids:
        raise RuntimeError("Tokenizer produced no token for PASS/FAIL — cannot read verdict logprobs")
    return sorted(set(pass_ids)), sorted(set(fail_ids))


def forced_verdict_prob(
    model: Any,
    tokenizer: Any,
    example_prompt: str,
    verdict_prefix: str,
    pass_ids: Sequence[int],
    fail_ids: Sequence[int],
    max_length: int,
) -> float:
    """P(PASS) на позиции вердикта при форсированном якоре оси.

    Якорь форсируется, а не генерируется: нас интересует распределение ровно на
    той позиции, с которой читает вероятность инференс. Softmax берётся по паре
    PASS/FAIL — та же нормировка, что в ``logprobs._pass_prob``.
    """
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device
    ids = tokenizer(example_prompt + verdict_prefix, add_special_tokens=False)["input_ids"]
    ids = ids[-max_length:]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    pass_lp = torch.logsumexp(log_probs[list(pass_ids)], dim=0)
    fail_lp = torch.logsumexp(log_probs[list(fail_ids)], dim=0)
    return float(torch.softmax(torch.stack([fail_lp, pass_lp]), dim=0)[1])


def score_held_out(
    model: Any,
    tokenizer: Any,
    samples: Sequence[RagSample],
    config: FtConfig,
) -> list[AxisProbs]:
    """Вероятности обеих осей для held-out части фолда (без генерации)."""
    from rag_reliability.methods.ft_judge.data import build_examples as build  # noqa: PLC0415
    from rag_reliability.methods.m3.axes import AXES, axis_anchor  # noqa: PLC0415

    pass_ids, fail_ids = verdict_token_ids(tokenizer)
    was_training = model.training
    model.eval()
    probs: list[AxisProbs] = []
    try:
        for sample in samples:
            axis_probs: dict[str, float] = {}
            for axis in AXES:
                example = build([sample], mode=config.mode, prompts_dir=config.prompts_dir, axes=(axis,))[0]
                prefix = _eval_prefix(model, tokenizer, example, config)
                axis_probs[axis] = forced_verdict_prob(
                    model,
                    tokenizer,
                    _prompt_text(tokenizer, example),
                    prefix + f"{axis_anchor(axis)}:",
                    pass_ids,
                    fail_ids,
                    config.max_length,
                )
            probs.append(
                AxisProbs(
                    p_faith=axis_probs["faithfulness"],
                    p_rel=axis_probs["relevance"],
                    method="logprobs",
                )
            )
    finally:
        if was_training:
            model.train()
    return probs


def _eval_prefix(model: Any, tokenizer: Any, example: JudgeExample, config: FtConfig) -> str:
    """Что стоит в завершении до строки вердикта.

    В режиме ``direct`` — ничего. В режиме ``marker`` строку ``MARKER:`` модель
    писала на обучении, поэтому её нужно получить от самой модели: подставить
    золотой маркер значило бы подсказать ответ, а пропустить строку — сместить
    позицию вердикта относительно обучающего формата.
    """
    if config.mode != "marker":
        return ""
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device
    ids = tokenizer(_prompt_text(tokenizer, example), add_special_tokens=False)["input_ids"]
    ids = ids[-config.max_length :]
    generated = model.generate(
        input_ids=torch.tensor([ids], dtype=torch.long, device=device),
        max_new_tokens=16,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    text = tokenizer.decode(generated[0][len(ids) :], skip_special_tokens=True)
    line = text.split("\n", 1)[0].strip()
    return f"{line}\n" if line.upper().startswith("MARKER") else "MARKER: none\n"


def _epoch_checkpoints(output_dir: Path) -> dict[int, Path]:
    """Сохранённые эпохи: ``epoch{N}`` -> путь. Основа ``--resume``."""
    if not output_dir.exists():
        return {}
    found: dict[int, Path] = {}
    for path in output_dir.iterdir():
        if path.is_dir() and path.name.startswith("epoch"):
            suffix = path.name.removeprefix("epoch")
            if suffix.isdigit():
                found[int(suffix)] = path
    return found


def _prune_checkpoints(output_dir: Path, keep: int) -> None:
    import shutil  # noqa: PLC0415

    checkpoints = _epoch_checkpoints(output_dir)
    for epoch in sorted(checkpoints)[:-keep] if keep < len(checkpoints) else []:
        shutil.rmtree(checkpoints[epoch], ignore_errors=True)
        logger.info("pruned checkpoint %s", checkpoints[epoch])


def build_peft_model(config: FtConfig) -> tuple[Any, Any]:
    """Модель и токенизатор: bf16, gradient checkpointing, LoRA поверх."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if config.tuning == "lora":
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415

        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer


def build_optimizer(model: Any, config: FtConfig) -> Any:
    """8-bit AdamW, если есть bitsandbytes; иначе обычный AdamW с предупреждением."""
    import torch  # noqa: PLC0415

    parameters = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb  # noqa: PLC0415

        return bnb.optim.AdamW8bit(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    except ImportError:
        logger.warning(
            "bitsandbytes is unavailable; falling back to torch AdamW "
            "(a few GB more optimizer state)"
        )
        return torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )


def example_weights(examples: Sequence[JudgeExample], pos_weight: float) -> list[float]:
    """Вес примера в лоссе: PASS домножается на ``pos_weight``, FAIL — единица."""
    return [pos_weight if example.label == 1 else 1.0 for example in examples]


def train_fold_transformers(request: FoldRequest) -> FoldOutcome:  # noqa: C901 - цикл обучения
    """Обучение одного фолда на transformers + peft.

    Ручной цикл вместо ``Trainer``: нужны вероятности held-out после *каждой*
    эпохи и повзвешенный по классу лосс — с ``Trainer`` и то, и другое пришлось
    бы протаскивать колбэками поверх его же цикла.
    """
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415
    from transformers import get_linear_schedule_with_warmup  # noqa: PLC0415

    config = request.config
    assert_vram(config)
    set_seed(config.seed + request.fold)

    model, tokenizer = build_peft_model(config)
    output_dir = Path(config.output_dir)
    start_epoch = 0
    if request.resume:
        start_epoch = _load_latest_checkpoint(model, output_dir)

    encoded = [
        encode_example(tokenizer, example, config.max_length)
        for example in request.train_examples
    ]
    weights = example_weights(request.train_examples, request.pos_weight)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    indexed = list(range(len(encoded)))

    loader = DataLoader(
        indexed,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda rows: (
            _collate([encoded[row] for row in rows], pad_id),
            [weights[row] for row in rows],
        ),
        generator=torch.Generator().manual_seed(config.seed + request.fold),
    )
    steps_per_epoch = max(1, math.ceil(len(loader) / config.grad_accum))
    total_steps = max(1, steps_per_epoch * config.n_epochs)
    optimizer = build_optimizer(model, config)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    device = next(model.parameters()).device
    probs: list[AxisProbs] = []
    model.train()
    for epoch in range(start_epoch + 1, config.n_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        for step, (batch, batch_weights) in enumerate(loader, start=1):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = _weighted_loss(
                outputs.logits, batch["labels"].to(device), batch_weights, device
            )
            (loss / config.grad_accum).backward()
            if step % config.grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        probs = score_held_out(model, tokenizer, request.held_out, config)
        request.on_epoch_end(epoch, probs)
        if config.save_strategy == "epoch":
            _save_checkpoint(model, tokenizer, output_dir, epoch)
            _prune_checkpoints(output_dir, config.save_total_limit)

    if not probs:
        # --resume на уже доученном прогоне: обучать нечего, но артефакт обязан
        # быть посчитан — иначе задание «успешно» завершится без скоров.
        probs = score_held_out(model, tokenizer, request.held_out, config)

    checkpoint = _save_checkpoint(model, tokenizer, output_dir, config.n_epochs)
    if config.push_to_hub:
        _push_to_hub(model, tokenizer, config.push_to_hub)
    return FoldOutcome(probs=tuple(probs), checkpoint=checkpoint)


def _weighted_loss(logits: Any, labels: Any, weights: Sequence[float], device: Any) -> Any:
    """Кросс-энтропия по токенам завершения, взвешенная по классу примера."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as functional  # noqa: PLC0415

    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    per_token = functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    mask = (shift_labels != -100).float()
    per_example = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    weight = torch.tensor(list(weights), dtype=per_example.dtype, device=device)
    return (per_example * weight).sum() / weight.sum().clamp(min=1e-9)


def _save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, epoch: int) -> str:
    """Адаптер и токенизатор в ``results/`` (в .gitignore); в run.yaml идёт путь."""
    path = output_dir / f"epoch{epoch}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    logger.info("saved checkpoint %s", path)
    return str(path)


def _load_latest_checkpoint(model: Any, output_dir: Path) -> int:
    """Догрузить последний сохранённый адаптер; вернуть номер доученной эпохи."""
    checkpoints = _epoch_checkpoints(output_dir)
    if not checkpoints:
        return 0
    epoch = max(checkpoints)
    path = checkpoints[epoch]
    if hasattr(model, "load_adapter"):
        model.load_adapter(str(path), adapter_name="default", is_trainable=True)
    else:
        raise RuntimeError(
            f"--resume found checkpoint {path}, but the model cannot load an adapter; "
            "full fine-tuning has no resume path here"
        )
    logger.info("resumed from %s (epoch %d already trained)", path, epoch)
    return epoch


def _push_to_hub(model: Any, tokenizer: Any, repo: str) -> None:
    logger.info("pushing adapter to the HF Hub: %s", repo)
    model.push_to_hub(repo)
    tokenizer.push_to_hub(repo)
