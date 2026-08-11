"""Артефакт энкодера: ``scores.jsonl`` с сырым логитом и инференс чекпоинта.

Метод отдаёт скор, а не решение. ``faithfulness_pred``/``relevance_pred``
остаются нулями намеренно: бинаризация — дело протокола (порог подбирается
внутри train-части фолда в ``evaluate_cv``), а прежний скрипт вшивал в артефакт
собственный порог, подобранный на своей же валидации.

Рядом с ``enc.logit`` пишется ``enc.prob`` — сигмоида того же логита. Причина
техническая: ``evaluate_cv`` требует скор в [0, 1], а язык ``--score-expr``
умеет только ``+ - * / ( )`` и сигмоиду выразить не может. Сырой логит остаётся
в артефакте как контрактный ключ метода и как вход стэкера C1.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_reliability.dataset import save_jsonl
from rag_reliability.schema import Prediction

if TYPE_CHECKING:
    from rag_reliability.methods.encoder.train import TrainConfig
    from rag_reliability.schema import RagSample

LOGIT_KEY = "enc.logit"
PROB_KEY = "enc.prob"
PROB_METHOD = "encoder_oof"
#: Кейсы вне folds.json: среднее по моделям фолдов. Помечены отдельно, потому
#: что ансамбль систематически чуть сильнее одиночной модели, и стэкер C1
#: должен видеть разницу, а не считать все строки однородными.
ENSEMBLE_PROB_METHOD = "encoder_fold_ensemble"


def sigmoid(logit: float) -> float:
    """Численно устойчивая сигмоида: exp(710) переполняет float и роняет валидацию."""
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


def logits_to_predictions(
    logits: dict[str, float], *, ensemble_ids: Collection[str] = ()
) -> list[Prediction]:
    """Логиты -> строки артефакта. Порядок словаря сохраняется.

    ``ensemble_ids`` — кейсы вне ``folds.json``, скоренные средним по моделям
    фолдов; в ``prob_method`` у них другой источник, чтобы строку с ансамблевым
    скором нельзя было спутать с честной out-of-fold.
    """
    if not logits:
        raise ValueError("Cannot build an artifact from an empty set of logits")
    unknown = [sample_id for sample_id in ensemble_ids if sample_id not in logits]
    if unknown:
        raise ValueError(
            f"{len(unknown)} ensemble id(s) have no logit: {unknown[:5]}"
        )
    ensemble = set(ensemble_ids)
    predictions: list[Prediction] = []
    for sample_id, logit in logits.items():
        value = float(logit)
        if not math.isfinite(value):
            raise ValueError(f"Logit for sample {sample_id!r} is not finite: {logit!r}")
        predictions.append(
            Prediction(
                id=sample_id,
                faithfulness_pred=0,
                relevance_pred=0,
                prob_method=ENSEMBLE_PROB_METHOD if sample_id in ensemble else PROB_METHOD,
                scores={LOGIT_KEY: value, PROB_KEY: sigmoid(value)},
            )
        )
    return predictions


def write_scores(predictions: Sequence[Prediction], path: str | Path) -> int:
    """Записать ``scores.jsonl`` и вернуть число строк."""
    save_jsonl(predictions, path)
    return len(predictions)


def predict_logits(
    samples: Sequence[RagSample],
    checkpoint: str | Path,
    config: TrainConfig,
) -> dict[str, float]:
    """Логиты готового чекпоинта по произвольному набору кейсов.

    Нужен для кейсов вне ``folds.json`` и для инференса на новых данных, где
    OOF-обучения не бывает. Тяжёлые импорты — внутри функции.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    from rag_reliability.methods.encoder.data import make_examples  # noqa: PLC0415
    from rag_reliability.methods.encoder.train import (  # noqa: PLC0415
        batched_logits,
        build_model,
    )

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {path.resolve()}")

    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    examples = make_examples(samples, tokenizer, config.max_length)

    model = build_model(config.model, pos_weight=1.0)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values = batched_logits(model, examples, pad_id, config.batch_size)
    return {example.id: float(value) for example, value in zip(examples, values, strict=True)}


def checkpoint_sha256(path: str | Path) -> str:
    """Хэш весов: в git они не едут, но прогон должен быть привязан к файлу."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_meta(checkpoints: dict[int, str]) -> list[dict[str, Any]]:
    """Пути и хэши чекпоинтов для ``run.yaml``; отсутствующий файл — не молчание."""
    meta: list[dict[str, Any]] = []
    for fold in sorted(checkpoints):
        path = Path(checkpoints[fold])
        meta.append(
            {
                "fold": fold,
                "path": str(path),
                "sha256": checkpoint_sha256(path) if path.exists() else None,
                "exists": path.exists(),
            }
        )
    return meta
