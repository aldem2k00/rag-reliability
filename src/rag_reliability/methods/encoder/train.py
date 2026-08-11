"""OOF-обучение энкодера по ``data/splits/folds.json`` с контролем схлопывания.

Два решения, ради которых модуль вообще появился.

**Сплит не создаётся здесь.** Прежний скрипт вызывал собственный
``train_test_split`` — четвёртый протокол разбиения в кодовой базе, к тому же
не group-aware. Разбиение читается из ``folds.json``; кейс скорится моделью,
которая его не видела, иначе стэкер C1 получит признак, видевший свою метку.

**Схлопывание ловится, а не обнаруживается задним числом.** После каждой эпохи
считается ``degenerate_rate`` по held-out части фолда, и прогон с
``const_share > 0.98`` помечается ``collapsed``. Вывод «длинный контекст вредит»
из ``docs/experiments.md`` был сделан ровно на таком прогоне: recall = 1.0000,
macro-F1 = 0.4191 — класс схлопнулся в «всё надёжно». Это провал обучения, а не
свойство длины, и такая конфигурация не имеет права считаться лучшей.

Стоимость. Полный 5×5 OOF — 25 обучений (~50 GPU-ч). Здесь 5 фолдов без
повторов (~10 GPU-ч), разброс оценивается бутстрэпом по кейсам; ``n_repeats: 1``
пишется в ``run.yaml`` явно, чтобы при сравнении с другими методами это было видно.

Торч импортируется лениво и только в реализации обучения фолда: сама раскладка
по фолдам, диагностика и сборка OOF тестируются на подставном тренере.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rag_reliability.metrics import degenerate_rate
from rag_reliability.schema import Prediction

if TYPE_CHECKING:
    from rag_reliability.methods.surface.oof import Folds
    from rag_reliability.schema import RagSample

logger = logging.getLogger(__name__)

POS_WEIGHT_MODES: tuple[str, ...] = ("none", "balanced")

#: Логит >= 0 <=> вероятность >= 0.5. Порог диагностики, не порог отчёта:
#: рабочий порог подбирается протоколом внутри train-части фолда.
DECISION_LOGIT = 0.0


@dataclass(frozen=True)
class TrainConfig:
    """Гиперпараметры одного прогона. Всё, что попадает в ``run.yaml``."""

    model: str = "deepvk/RuModernBERT-base"
    max_length: int = 512
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.06
    batch_size: int = 4
    grad_accum: int = 1
    epochs: float = 3.0
    pos_weight_mode: str = "none"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    output_dir: str = "results/encoder_checkpoints"

    def __post_init__(self) -> None:
        if self.pos_weight_mode not in POS_WEIGHT_MODES:
            raise ValueError(
                f"pos_weight_mode must be one of {POS_WEIGHT_MODES}, got {self.pos_weight_mode!r}"
            )
        for name in ("max_length", "batch_size", "grad_accum"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")

    @property
    def n_epochs(self) -> int:
        """Число эпох, после каждой из которых обязана быть диагностика."""
        return math.ceil(self.epochs)


@dataclass(frozen=True)
class EpochLog:
    """Диагностика одной эпохи одного фолда."""

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


EpochHook = Callable[[int, Sequence[float]], EpochLog]


@dataclass(frozen=True)
class FoldRequest:
    """Всё, что нужно обучению одного фолда, включая обязательный хук эпохи."""

    repeat: int
    fold: int
    train_samples: tuple[RagSample, ...]
    test_samples: tuple[RagSample, ...]
    config: TrainConfig
    on_epoch_end: EpochHook
    #: Кейсы вне ``folds.json`` — их скорит каждая модель фолда, см. ``train_oof_detailed``.
    extra_samples: tuple[RagSample, ...] = ()


@dataclass(frozen=True)
class FoldOutcome:
    """Результат обучения фолда: логиты held-out части и путь к чекпоинту."""

    logits: tuple[float, ...]
    checkpoint: str | None = None
    #: Логиты ``extra_samples`` этой же моделью; усредняются по фолдам.
    extra_logits: tuple[float, ...] = ()


FoldTrainer = Callable[[FoldRequest], FoldOutcome]


@dataclass
class OofResult:
    """Логиты по всему корпусу плюс всё, без чего прогон нельзя честно описать.

    Два источника скора, и их нельзя путать. ``oof_ids`` предсказаны честным
    out-of-fold: модель фолда не видела ни одного кейса своего фолда.
    ``ensemble_ids`` — кейсы вне ``folds.json``; их скорит среднее по всем
    моделям фолдов. Это не ухудшение изоляции, а её усиление: вне фолдов лежит
    ровно одна oversized-группа, целиком отсутствующая в train-части любого
    фолда, так что модель не видела не только сам кейс, но и всю его группу.
    Ансамблевое усреднение при этом даёт им чуть более сильный скор, чем
    одиночная модель, — поэтому источник помечен в артефакте и в ``run.yaml``,
    а метрики по ним не считаются: ``evaluate_cv`` их отбрасывает по отсутствию
    фолда, и вердикт о схлопывании тоже считается только по ``oof_ids``.
    """

    logits: dict[str, float]
    oof_ids: tuple[str, ...] = ()
    ensemble_ids: tuple[str, ...] = ()
    epochs: list[EpochLog] = field(default_factory=list)
    checkpoints: dict[int, str] = field(default_factory=dict)
    repeat: int = 0
    n_repeats: int = 1

    @property
    def oof_logits(self) -> dict[str, float]:
        """Только честный out-of-fold: на нём меряют качество и схлопывание."""
        if not self.oof_ids:
            return dict(self.logits)
        return {sample_id: self.logits[sample_id] for sample_id in self.oof_ids}

    @property
    def collapsed(self) -> bool:
        return self.collapse_reason is not None

    @property
    def collapsed_folds(self) -> list[int]:
        """Фолды, чья модель на последней эпохе выдала один класс на всём held-out."""
        last: dict[int, EpochLog] = {}
        for log in self.epochs:
            previous = last.get(log.fold)
            if previous is None or log.epoch >= previous.epoch:
                last[log.fold] = log
        return sorted(fold for fold, log in last.items() if log.is_degenerate)

    @property
    def collapse_reason(self) -> str | None:
        """Почему прогон считается схлопнувшимся — или ``None``, если не считается.

        Одного взгляда на склеенный OOF мало. Модели фолдов инициализируются
        по-разному, и прогон, где каждый фолд предсказал ровно один класс, но
        разные фолды — разные классы, даёт вполне приличный общий ``const_share``.
        Обучения при этом не было ни в одном фолде. Смоук на крошечной модели
        воспроизвёл ровно этот случай: 3 фолда по const_share = 1.0, пул 0.67.
        """
        if is_collapsed(self.oof_logits):
            return "pooled"
        trained = {log.fold for log in self.epochs}
        if trained and len(self.collapsed_folds) == len(trained):
            return "per_fold"
        return None

    def diagnostics(self) -> dict[str, Any]:
        summary = degenerate_rate(decisions_from_logits(self.oof_logits))
        return {
            "n_scored": len(self.logits),
            "n_oof": len(self.oof_logits),
            "n_ensemble": len(self.ensemble_ids),
            "repeat": self.repeat,
            "n_repeats": self.n_repeats,
            "collapsed": self.collapsed,
            "collapse_reason": self.collapse_reason,
            "collapsed_folds": self.collapsed_folds,
            "const_share": float(summary["const_share"]),
            "output_entropy": float(summary["output_entropy"]),
            "epochs": [log.as_dict() for log in self.epochs],
        }


def decisions_from_logits(logits: dict[str, float]) -> list[Prediction]:
    """Бинарные решения ради диагностики схлопывания.

    Артефакт метода бинарных решений не содержит — их принимает протокол. Но
    ``degenerate_rate`` по определению смотрит на решения, и на артефактных
    нулях он показал бы схлопывание у любого прогона. Поэтому здесь логит
    сравнивается с нулём: это диагностический порог, а не отчётный.
    """
    if not logits:
        raise ValueError("Cannot diagnose an empty set of logits")
    decisions: list[Prediction] = []
    for sample_id, logit in logits.items():
        label = int(logit >= DECISION_LOGIT)
        decisions.append(
            Prediction(id=sample_id, faithfulness_pred=label, relevance_pred=label)
        )
    return decisions


def is_collapsed(logits: dict[str, float]) -> bool:
    """Прогон схлопнулся, если почти все решения одинаковы (порог — из metrics.py)."""
    return bool(degenerate_rate(decisions_from_logits(logits))["is_degenerate"])


def compute_pos_weight(labels: Sequence[int], mode: str) -> float:
    """``n_neg / n_pos`` по train-части фолда; ``none`` — единица."""
    if mode not in POS_WEIGHT_MODES:
        raise ValueError(f"pos_weight_mode must be one of {POS_WEIGHT_MODES}, got {mode!r}")
    if mode == "none":
        return 1.0
    positives = sum(labels)
    return (len(labels) - positives) / max(positives, 1)


def make_epoch_hook(*, repeat: int, fold: int, sink: list[EpochLog]) -> EpochHook:
    """Хук, который считает ``degenerate_rate`` и пишет его в лог и в ``sink``."""

    def on_epoch_end(epoch: int, logits: Sequence[float]) -> EpochLog:
        if not logits:
            raise ValueError(
                f"repeat {repeat} fold {fold} epoch {epoch}: no held-out logits to diagnose"
            )
        indexed = {str(index): float(value) for index, value in enumerate(logits)}
        summary = degenerate_rate(decisions_from_logits(indexed))
        log = EpochLog(
            repeat=repeat,
            fold=fold,
            epoch=epoch,
            n_held_out=len(logits),
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


def _fold_partition(
    samples: Sequence[RagSample], folds: Folds, *, repeat: int, fold: int
) -> tuple[tuple[RagSample, ...], tuple[RagSample, ...]]:
    train: list[RagSample] = []
    test: list[RagSample] = []
    for sample in samples:
        assignment = folds.assignment[sample.id]
        (test if assignment[repeat] == fold else train).append(sample)
    return tuple(train), tuple(test)


def _split_by_fold_assignment(
    samples: Sequence[RagSample], folds: Folds
) -> tuple[tuple[RagSample, ...], tuple[RagSample, ...]]:
    """Кейсы с номером фолда и кейсы без него; порядок корпуса сохраняется."""
    inside = tuple(sample for sample in samples if sample.id in folds.assignment)
    outside = tuple(sample for sample in samples if sample.id not in folds.assignment)
    return inside, outside


def train_oof(
    samples: Sequence[RagSample],
    folds: Folds,
    config: TrainConfig,
    *,
    repeat: int = 0,
    train_fold: FoldTrainer | None = None,
) -> dict[str, float]:
    """Обучает n_folds моделей, каждая предсказывает свой held-out фолд.

    Возвращает OOF-логиты по всем кейсам -> ``scores['enc.logit']``. «По всем» —
    буквально: кейсы, которым ``folds.json`` не дал фолда, скорит среднее по
    моделям фолдов (см. ``OofResult``), иначе артефакт метода покрывал бы две
    трети корпуса.
    """
    return train_oof_detailed(
        samples, folds, config, repeat=repeat, train_fold=train_fold
    ).logits


def train_oof_detailed(
    samples: Sequence[RagSample],
    folds: Folds,
    config: TrainConfig,
    *,
    repeat: int = 0,
    train_fold: FoldTrainer | None = None,
) -> OofResult:
    """То же, что ``train_oof``, но с диагностикой, нужной ``run.yaml``.

    ``train_fold`` вынесен в параметр не ради гибкости, а ради тестируемости:
    раскладка по фолдам и контроль схлопывания обязаны проверяться без GPU.
    """
    if not samples:
        raise ValueError("Cannot train on an empty corpus")
    if not 0 <= repeat < folds.n_repeats:
        raise ValueError(
            f"repeat must be in [0, {folds.n_repeats}), got {repeat}; "
            "folds.json declares fewer repeats than requested"
        )
    inside, outside = _split_by_fold_assignment(samples, folds)
    if not inside:
        raise ValueError(
            f"None of the {len(samples)} sample(s) have a fold assignment; "
            "wrong corpus or wrong folds file"
        )

    trainer = train_fold if train_fold is not None else train_fold_transformers
    result = OofResult(logits={}, repeat=repeat, n_repeats=1)
    outside_totals: dict[str, float] = {sample.id: 0.0 for sample in outside}
    outside_counts: dict[str, int] = {sample.id: 0 for sample in outside}

    for fold in range(folds.n_folds):
        train_samples, test_samples = _fold_partition(inside, folds, repeat=repeat, fold=fold)
        if not test_samples:
            # На полном корпусе пустых фолдов не бывает; они появляются только под
            # --limit. Обучать модель, которой некого предсказывать, незачем —
            # но и молчать нельзя. Полноту покрытия ловит проверка never_scored.
            logger.warning(
                "repeat %d fold %d has no held-out case in this run and is skipped", repeat, fold
            )
            continue
        if not train_samples:
            raise ValueError(f"repeat {repeat} fold {fold} leaves no training part")

        before = len(result.epochs)
        outcome = trainer(
            FoldRequest(
                repeat=repeat,
                fold=fold,
                train_samples=train_samples,
                test_samples=test_samples,
                config=config,
                on_epoch_end=make_epoch_hook(repeat=repeat, fold=fold, sink=result.epochs),
                extra_samples=outside,
            )
        )
        if len(outcome.logits) != len(test_samples):
            raise ValueError(
                f"repeat {repeat} fold {fold}: trainer returned {len(outcome.logits)} logit(s) "
                f"for {len(test_samples)} held-out case(s)"
            )
        if len(outcome.extra_logits) != len(outside):
            raise ValueError(
                f"repeat {repeat} fold {fold}: trainer returned {len(outcome.extra_logits)} "
                f"logit(s) for {len(outside)} case(s) outside the folds; the artifact must "
                "cover the whole corpus"
            )
        logged = len(result.epochs) - before
        if logged != config.n_epochs:
            raise ValueError(
                f"repeat {repeat} fold {fold}: collapse diagnostics ran {logged} time(s) for "
                f"{config.n_epochs} epoch(s); degenerate_rate is mandatory after every epoch"
            )
        for sample, logit in zip(test_samples, outcome.logits, strict=True):
            result.logits[sample.id] = float(logit)
        for sample, logit in zip(outside, outcome.extra_logits, strict=True):
            outside_totals[sample.id] += float(logit)
            outside_counts[sample.id] += 1
        if outcome.checkpoint is not None:
            result.checkpoints[fold] = outcome.checkpoint

    result.oof_ids = tuple(result.logits)
    never_scored = [sample.id for sample in inside if sample.id not in result.logits]
    if never_scored:
        raise ValueError(
            f"{len(never_scored)} sample(s) never landed in a held-out fold: {never_scored[:5]}"
        )

    unscored_outside = [sample_id for sample_id, count in outside_counts.items() if count == 0]
    if unscored_outside:
        raise ValueError(
            f"{len(unscored_outside)} case(s) outside the folds were never scored: "
            f"{unscored_outside[:5]}; no fold model ran"
        )
    for sample in outside:
        result.logits[sample.id] = outside_totals[sample.id] / outside_counts[sample.id]
    result.ensemble_ids = tuple(sample.id for sample in outside)
    if outside:
        logger.info(
            "%d case(s) outside %d-fold assignment scored by the mean of %d fold model(s)",
            len(outside),
            folds.n_folds,
            max(outside_counts.values()),
        )

    if result.collapsed:
        logger.warning(
            "run collapsed (%s): const_share=%.4f, collapsed folds %s — configuration is "
            "excluded from best-config choice",
            result.collapse_reason,
            float(degenerate_rate(decisions_from_logits(result.oof_logits))["const_share"]),
            result.collapsed_folds,
        )
    return result


# --------------------------------------------------------------------------- #
# Реализация обучения фолда на transformers. Всё тяжёлое — только здесь.
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
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(model_name: str, pos_weight: float) -> Any:
    """Mean-pool голова над энкодером — та же, на которой получено 0.5879."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from transformers import AutoModel  # noqa: PLC0415

    def mean_pool(hidden: Any, mask: Any) -> Any:
        expanded_mask = mask.unsqueeze(-1).float()
        return (hidden * expanded_mask).sum(1) / expanded_mask.sum(1).clamp(min=1e-9)

    class Classifier(nn.Module):
        def __init__(self, name: str, weight: float) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(name, trust_remote_code=True)
            self.head = nn.Sequential(
                nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, 1)
            )
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None) -> Any:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.head(mean_pool(output.last_hidden_state, attention_mask)).squeeze(-1)
            loss = self.loss_fn(logits, labels.float()) if labels is not None else None
            return {"loss": loss, "logits": logits}

    return Classifier(model_name, pos_weight)


def _collate(batch: list[Any], pad_id: int) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    width = max(len(item.input_ids) for item in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.tensor([item.label for item in batch], dtype=torch.long)
    for row, item in enumerate(batch):
        input_ids[row, : len(item.input_ids)] = torch.tensor(item.input_ids, dtype=torch.long)
        attention_mask[row, : len(item.input_ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def batched_logits(model: Any, examples: Sequence[Any], pad_id: int, batch_size: int) -> list[float]:
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device
    model.eval()
    output: list[float] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = _collate(list(examples[start : start + batch_size]), pad_id)
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )["logits"]
            output.extend(float(value) for value in logits.detach().cpu().reshape(-1))
    model.train()
    return output


def train_fold_transformers(request: FoldRequest) -> FoldOutcome:
    """Обучение одного фолда на torch/transformers.

    Ручной цикл вместо ``Trainer``: нужны логиты held-out после *каждой* эпохи —
    без них контроль схлопывания превращается в постфактум-объяснение.
    """
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: PLC0415

    from rag_reliability.methods.encoder.data import make_examples  # noqa: PLC0415

    config = request.config
    set_seed(config.seed + request.fold)
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    train_examples = make_examples(request.train_samples, tokenizer, config.max_length)
    test_examples = make_examples(request.test_samples, tokenizer, config.max_length)
    extra_examples = make_examples(request.extra_samples, tokenizer, config.max_length)
    pos_weight = compute_pos_weight(
        [example.label for example in train_examples], config.pos_weight_mode
    )

    model = build_model(config.model, pos_weight)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    loader = DataLoader(
        train_examples,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, pad_id),
        generator=torch.Generator().manual_seed(config.seed + request.fold),
    )
    steps_per_epoch = max(1, math.ceil(len(loader) / config.grad_accum))
    total_steps = max(1, steps_per_epoch * config.n_epochs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    logits: list[float] = []
    for epoch in range(1, config.n_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            (outputs["loss"] / config.grad_accum).backward()
            if step % config.grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        # OOF-логиты — это логиты последней эпохи: второй прогон инференса ради
        # того же числа стоил бы лишний проход по held-out на каждом фолде.
        logits = batched_logits(model, test_examples, pad_id, config.batch_size)
        request.on_epoch_end(epoch, logits)

    extra = (
        batched_logits(model, extra_examples, pad_id, config.batch_size) if extra_examples else []
    )
    checkpoint = _save_checkpoint(model, config, request.repeat, request.fold)
    return FoldOutcome(
        logits=tuple(logits), checkpoint=checkpoint, extra_logits=tuple(extra)
    )


def _save_checkpoint(model: Any, config: TrainConfig, repeat: int, fold: int) -> str:
    """Веса кладутся в ``results/`` (в .gitignore); в run.yaml идёт путь и хэш."""
    from pathlib import Path  # noqa: PLC0415

    import torch  # noqa: PLC0415

    directory = Path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"fold{fold}_repeat{repeat}.pt"
    torch.save(model.state_dict(), path)
    return str(path)
