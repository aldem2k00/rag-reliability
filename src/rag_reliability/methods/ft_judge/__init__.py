"""Fine-tuning судьи Метода 3 по фолдам ``folds.json``.

Публичная поверхность пакета — сборка обучающих примеров, обучение одного
фолда с обязательной диагностикой схлопывания и сборка артефакта общего
контракта. Тяжёлые зависимости (torch, transformers, peft, bitsandbytes)
импортируются лениво внутри реализации обучения: раскладка по фолдам,
симметрия формата и диагностика обязаны проверяться без GPU.
"""

from __future__ import annotations

from rag_reliability.methods.ft_judge.data import (
    MODES,
    JudgeExample,
    build_examples,
    check_format_symmetry,
    class_balance,
    completion_text,
    compute_pos_weight,
    oversample_negatives,
)
from rag_reliability.methods.ft_judge.predict import (
    AxisProbs,
    probs_to_predictions,
    write_scores,
)
from rag_reliability.methods.ft_judge.train import (
    EpochLog,
    FoldOutcome,
    FoldRequest,
    FoldResult,
    FtConfig,
    make_epoch_hook,
    train_one_fold,
)

__all__ = [
    "MODES",
    "AxisProbs",
    "EpochLog",
    "FoldOutcome",
    "FoldRequest",
    "FoldResult",
    "FtConfig",
    "JudgeExample",
    "build_examples",
    "check_format_symmetry",
    "class_balance",
    "completion_text",
    "compute_pos_weight",
    "make_epoch_hook",
    "oversample_negatives",
    "probs_to_predictions",
    "train_one_fold",
    "write_scores",
]
