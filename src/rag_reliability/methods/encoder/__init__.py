"""Энкодер-классификатор надёжности: построение входа, OOF-обучение, инференс.

Пакет существует потому, что логика жила в ``scripts/train_encoder_baseline.py``
и была непроверяемой: собственный сплит, отсутствие контроля схлопывания и
формат входа, отличный от того, на котором получены опубликованные числа.
Здесь всё это разнесено на тестируемые части, а скрипт остаётся обёрткой.
"""

from __future__ import annotations

from rag_reliability.methods.encoder.data import (
    EncodedInput,
    EncoderExample,
    EncoderSegments,
    build_encoder_text,
    build_segments,
    encode,
    make_examples,
    parse_chunks,
    split_dialog,
)
from rag_reliability.methods.encoder.predict import (
    LOGIT_KEY,
    PROB_KEY,
    checkpoint_meta,
    checkpoint_sha256,
    logits_to_predictions,
    predict_logits,
    sigmoid,
    write_scores,
)
from rag_reliability.methods.encoder.train import (
    EpochLog,
    FoldOutcome,
    FoldRequest,
    OofResult,
    TrainConfig,
    is_collapsed,
    train_oof,
    train_oof_detailed,
)

__all__ = [
    "LOGIT_KEY",
    "PROB_KEY",
    "EncodedInput",
    "EncoderExample",
    "EncoderSegments",
    "EpochLog",
    "FoldOutcome",
    "FoldRequest",
    "OofResult",
    "TrainConfig",
    "build_encoder_text",
    "build_segments",
    "checkpoint_meta",
    "checkpoint_sha256",
    "encode",
    "is_collapsed",
    "logits_to_predictions",
    "make_examples",
    "parse_chunks",
    "predict_logits",
    "sigmoid",
    "split_dialog",
    "train_oof",
    "train_oof_detailed",
    "write_scores",
]
