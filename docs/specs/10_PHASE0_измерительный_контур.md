# 10 — Фаза 0: измерительный контур

**Цель:** сделать числа проекта измеримыми и защитимыми. Прироста метрики фаза не даёт и не должна.
**GPU:** не нужен. **Оценка:** 5 рабочих дней. **Блокирует:** всё остальное.

---

## 0. Обоснование

Три факта, установленных в аудите:

- разброс единичного holdout при перетасовке сплита: sd = 0.032, размах 0.144 — вдвое больше максимальной разницы между методами в проекте (0.068);
- систематический оптимизм val-подобранного порога: +0.035…+0.053;
- нулевая калибровка: чистый шум через ту же процедуру даёт test-p95 = 0.5488, val-p95 = 0.5947 — внутри этого диапазона лежат все val-числа M3.

Пока это не исправлено, любой эксперимент производит числа, которые придётся пересчитывать.

---

## 1. Задача 0.1 — канонические сплиты

### 1.1 Что не так сейчас

`scripts/prepare_splits.py:24` вызывает `split_samples` из `dataset.py:42` — стратифицированное разбиение без учёта групп. Под ним:

| Утечка в test | Доля |
|---|---|
| тот же `full_dialog` целиком | 11.1% |
| нормализованный вопрос клиента | 24.9% |
| near-duplicate диалог (char-TF-IDF cos ≥ 0.99) | 13.3% |
| тот же `chunk_1` | 59.1% |

1-NN-запоминатель на char-TF-IDF под этим протоколом даёт **0.6278** — выше всех опубликованных чисел проекта.

### 1.2 Новый модуль `src/rag_reliability/splits.py`

```python
"""Group-aware стратифицированное разбиение корпуса на фолды.

Единственный источник истины по разбиению — data/splits/folds.json.
Инференс НЕ знает о сплитах; сплит применяется только на этапе оценки.
"""
from __future__ import annotations
import json, re, unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

from rag_reliability.schema import RagSample

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_query(text: str) -> str:
    """Нормализация реплики клиента для группировки: регистр, пунктуация, пробелы."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def build_groups(
    samples: list[RagSample],
    *,
    near_dup_threshold: float = 0.90,
    use_chunk_key: bool = True,
) -> dict[str, str]:
    """id -> group_key. Группа объединяет кейсы, которые нельзя разводить по фолдам.

    Три уровня склейки, применяются транзитивно:
      1) одинаковый нормализованный вопрос клиента;
      2) near-duplicate диалога (char 3-5 gram TF-IDF, косинус >= threshold);
      3) одинаковый chunk_1 (одна статья базы знаний), если use_chunk_key.

    Уровень 3 обязателен: без него 67% тестовых строк делят статью с train.
    """
    ...
```

**Алгоритм группировки** (реализация — union-find поверх трёх источников рёбер):

1. Ребро между `i` и `j`, если `normalize_query(client_turn_i) == normalize_query(client_turn_j)`.
2. Ребро, если `cosine(tfidf_char35(dialog_i), tfidf_char35(dialog_j)) ≥ 0.90`. Считать по разреженной матрице, брать только пары выше порога (`sklearn.metrics.pairwise.cosine_similarity` кусками по 500 строк, чтобы не держать 2233² в памяти).
3. Ребро, если `chunk_1_i == chunk_1_j`.

**Обработка вырожденных случаев.** Крупнейшая группа по уровню 1 — `оператор`, 136 строк (6.1% корпуса). Уровень 3 склеит ещё больше. Если после склейки самая крупная группа превышает 15% корпуса, она **не разбивается**, но помечается флагом `oversized: true` и целиком уходит в train во всех фолдах, а её вклад исключается из метрики (иначе один фолд станет несравнимым с остальными). Число исключённых кейсов пишется в `folds.json` и обязано фигурировать в отчёте.

### 1.3 Стратифицированное распределение групп по фолдам

Проблема протокола B (`m3m6/.../make_splits.py:57`): жадное распределение по размеру дало base rate `reliable` 72.4% / 75.3% / **67.9%** по train/val/test — разрыв 7.5 п.п., из-за которого порог с val не переносится на test.

Решение — **greedy bin-packing с балансировкой доли позитивов**:

```python
def assign_folds(
    samples: list[RagSample],
    groups: dict[str, str],
    *,
    n_folds: int = 5,
    n_repeats: int = 5,
    seed: int = 2233,
) -> dict[str, list[int]]:
    """id -> [fold_repeat_0, ..., fold_repeat_4].

    Для каждого повтора: группы сортируются по убыванию размера (детерминированно,
    tie-break по хэшу group_key с солью seed+repeat), затем каждая группа кладётся
    в фолд, минимизирующий взвешенную невязку:
        cost(fold) = |n_fold - n_target| / n_target
                   + LAMBDA * |pos_rate_fold - pos_rate_global|
    LAMBDA = 5.0 — стратификация важнее равенства размеров.
    """
```

**Допуск на приёмке:** доля `reliable` в каждом фолде отклоняется от глобальных 72.25% не более чем на **±2 п.п.**; размер фолда — не более чем на ±5% от 2233/5.

### 1.4 Артефакт `data/splits/folds.json`

```json
{
  "schema_version": 1,
  "corpus": {"path": "data/organizers.jsonl", "sha256": "...", "n": 2233},
  "config": {"n_folds": 5, "n_repeats": 5, "seed": 2233,
             "near_dup_threshold": 0.90, "use_chunk_key": true},
  "stats": {
    "n_groups": 1487,
    "largest_group": 136,
    "oversized_groups": ["grp_operator"],
    "excluded_ids": 136,
    "pos_rate_global": 0.7225,
    "pos_rate_by_fold": [[0.719, 0.731, 0.715, 0.728, 0.720], ...],
    "leak_check": {"query_overlap": 0.0, "near_dup_0.9": 0.008, "chunk1_overlap": 0.021}
  },
  "assignment": {"alfa_9b24723685d8": [0, 3, 1, 4, 2], ...}
}
```

Файл **коммитится**. Все скрипты читают его; ни один не пересплитывает самостоятельно.

### 1.5 Проверка `prepare_splits.py --check`

Отдельный режим, который не генерирует, а валидирует существующий `folds.json`:

| Проверка | Порог | Действие при провале |
|---|---|---|
| sha256 корпуса совпадает | точно | ошибка |
| утечка по нормализованному вопросу между фолдами | = 0% | ошибка |
| near-duplicate (cos ≥ 0.9) между фолдами | < 2% | ошибка |
| `chunk_1` между фолдами | < 5% | предупреждение |
| отклонение base rate по фолдам | ≤ 2 п.п. | ошибка |
| отклонение размера фолда | ≤ 5% | предупреждение |

Добавить в CI (`.github/workflows/ci.yml`).

### 1.6 Регресс-тест: цена утечки

`tests/test_splits_leakage.py` — воспроизводит и фиксирует главный аргумент в пользу протокола:

```python
def test_memorizer_gap_between_protocols():
    """1-NN на char-TF-IDF: стратифицированный сплит >> group-aware.

    Это не тест кода, а зафиксированное измерение, которое идёт в статью:
    протокол A даёт запоминателю 0.63, протокол B — 0.51.
    """
    assert nn_macro_f1(stratified_split) > 0.60
    assert nn_macro_f1(group_aware_folds) < 0.55
```

---

## 2. Задача 0.2 — расширение контракта предсказания

### 2.1 Изменение `schema.py`

```python
class Prediction(BaseModel):
    ...  # существующие поля без изменений
    scores: dict[str, float] = Field(default_factory=dict)

    @field_validator("scores")
    @classmethod
    def _score_keys(cls, v: dict[str, float]) -> dict[str, float]:
        bad = [k for k in v if "." not in k]
        if bad:
            raise ValueError(f"score keys must be '<method>.<signal>', got {bad[:5]}")
        return v
```

Соглашение об именовании обязательно: стэкер (`stacking/collect.py`) собирает фичи по префиксу и падает на коллизиях.

### 2.2 Миграция существующих артефактов

Скрипт `scripts/migrate_predictions.py` — одноразовый:

- читает старые `val.jsonl`/`test.jsonl`;
- переносит `p_faith`/`p_rel` в `scores` как `<method>.p_faith`/`<method>.p_rel`, сохраняя исходные поля;
- склеивает val+test в один `scores.jsonl`;
- **не** пытается достроить недостающие 1787 строк — те, кто их не имеет, помечаются `partial: true` в `run.yaml` и не участвуют в CV до полного прогона.

---

## 3. Задача 0.3 — пакет `evaluation/`

### 3.1 `evaluation/protocol.py` — ядро

```python
@dataclass(frozen=True)
class CVResult:
    oof_scores: np.ndarray          # (n,) итоговый скор надёжности, out-of-fold
    oof_pred: np.ndarray            # (n,) бинарное решение при OOF-пороге
    y: np.ndarray                   # (n,) золото
    per_repeat_f1: list[float]      # 5 значений
    thresholds: list[float]         # порог, подобранный в каждом фолде
    n_excluded: int


def evaluate_cv(
    samples, predictions, folds, *,
    score_fn=lambda p: (p.scores.get("p_faith", ...) * p.scores.get("p_rel", ...)),
    fit_fn=None,          # None -> только порог; иначе обучаемая модель (стэк, энкодер)
    grid_step=0.01,
) -> CVResult:
    """Оценка по фолдам с вложенным подбором порога.

    Для repeat r, fold k:
        train_idx = все, кроме фолда k              -> fit_fn (если задан) + подбор порога
        test_idx  = фолд k                          -> применение
    OOF-предсказания склеиваются; метрика считается один раз по всем 2233.
    """
```

**Ключевое требование:** порог подбирается **только** на train-части фолда. Никогда — на всём корпусе, никогда — на held-out.

### 3.2 `evaluation/metrics.py` — что добавить

Существующий `evaluate_predictions` (`metrics.py:34`) сохраняется как есть — это контракт платформы. Добавляются:

```python
def operational_metrics(y_true, scores, threshold) -> dict:
    """Метрики риск-гейта. Позитивный класс = unreliable."""
    return {
        "roc_auc": ...,
        "pr_auc_unreliable": ...,          # + base_rate и lift = pr_auc / base_rate
        "base_rate_unreliable": ...,
        "lift": ...,
        "recall_at_precision": {0.5: ..., 0.6: ..., 0.7: ...},
        "flagged_share": ...,              # доля помеченного трафика
        "confusion": {"tn":..., "fp":..., "fn":..., "tp":...},
    }


def degenerate_rate(predictions) -> dict:
    """Диагностика вырожденных выходов — то, чего invalid_output_rate не ловит.

    invalid_output_rate = 0 достигается константой (1,1): проверено, majority даёт 0.0.
    """
    return {
        "const_share": ...,        # доля самого частого (faith,rel) сочетания
        "output_entropy": ...,     # энтропия распределения по 4 сочетаниям
        "is_degenerate": ...,      # const_share > 0.98
    }
```

### 3.3 `evaluation/bootstrap.py`

```python
def bootstrap_ci(y, pred, metric_fn, *, B=10_000, seed=0) -> tuple[float, float, float]:
    """Точечная оценка и 95% ДИ перцентильным бутстрэпом по кейсам."""

def paired_bootstrap(y, pred_a, pred_b, metric_fn, *, B=10_000, seed=0) -> dict:
    """Δ = metric(a) - metric(b), 95% ДИ разности, двусторонний p."""

def exact_mcnemar(y, pred_a, pred_b) -> dict:
    """b, c и точный биномиальный p по построчной корректности."""
```

Ожидаемые числа для регресс-теста (на текущих 223-строчных артефактах): `surface vs m3_zero_shot` → Δ = +0.0141, ДИ [−0.076, +0.105], p = 0.765; McNemar b=31, c=27, p = 0.694.

### 3.4 `evaluation/nullcal.py` — нулевая калибровка

```python
def null_calibration(y, folds, *, n_trials=500, grid_step=0.01, seed=0) -> dict:
    """Что выжимает из процедуры подбора порога чистый шум.

    Скоры ~ U(0,1), метки реальные, процедура — та же (fit порога на train-фолдах,
    применение к held-out). Возвращает распределение macro-F1: mean, p50, p90, p95, p99.

    Печатается рядом с КАЖДЫМ числом в отчёте: 'перцентиль нулевого распределения'.
    """
```

Это самая дешёвая и самая полезная часть фазы: она автоматически помечает результаты, неотличимые от шума.

---

## 4. Задача 0.4 — единая точка оценки

### 4.1 `scripts/evaluate_cv.py`

```bash
python scripts/evaluate_cv.py \
  --data data/organizers.jsonl \
  --folds data/splits/folds.json \
  --scores predictions/m3_judge/zero_shot/scores.jsonl \
  --score-expr "m3.p_faith * m3.p_rel" \
  --output predictions/m3_judge/zero_shot/report.json
```

### 4.2 Схема `report.json`

```json
{
  "schema_version": 1,
  "method": "m3_judge", "variant": "zero_shot",
  "protocol": {"folds": "data/splits/folds.json", "sha256": "...",
               "n_folds": 5, "n_repeats": 5, "n_evaluated": 2097, "n_excluded": 136},
  "primary": {
    "reliable_f1_macro": 0.5xx,
    "ci95": [0.5xx, 0.5xx],
    "per_repeat": [...],
    "sd_across_repeats": 0.00x,
    "null_percentile": 76.0,
    "above_noise": false
  },
  "axes": {"faithfulness_f1_macro": {...}, "relevance_f1_macro": {...}},
  "operational": {"roc_auc": ..., "pr_auc_unreliable": ..., "lift": ...,
                  "recall_at_precision": {...}, "flagged_share": ...},
  "diagnostics": {"degenerate": {...}, "invalid_output_rate": 0.0,
                  "thresholds_per_fold": [...], "threshold_sd": 0.0x},
  "comparisons": [
    {"vs": "baselines/surface", "delta": 0.0xx, "ci95": [...], "p": 0.xx, "significant": false}
  ]
}
```

**Валидация:** `report.json` без `ci95` и без `null_percentile` не собирается — pydantic-модель `EvaluationReport` с обязательными полями.

### 4.3 Регресс-тест на старых числах

`tests/test_protocol_regression.py`: `evaluate_cv` в режиме «единичный holdout + 2 порога» должен воспроизводить опубликованные 10 значений до 1e-6. Это гарантирует, что новый контур не сломал арифметику, а изменил только протокол.

---

## 5. Задача 0.5 — гигиена репозитория

| # | Действие | Файл |
|---|---|---|
| 5.1 | `git_hash()` → `(hash, dirty, changed_files)`; в `run.yaml` пишется `dirty: true` | `run_meta.py:31` |
| 5.2 | Перенести `predictions/cloud/**` → `predictions/pseudo_debug/**`; обновить `predictions/README.md` | — |
| 5.3 | Исправить `summary_table.md:1`: «full (2245)» → «full (2233)» | `results/summary_table.md` |
| 5.4 | Убрать из `README.md:196-198` таблицу, смешивающую in-sample и held-out числа; заменить на ссылку на CV-отчёты | `README.md` |
| 5.5 | Починить `dvc.lock` под раскладку `src/rag_reliability/` **или** удалить DVC целиком | `dvc.yaml`, `dvc.lock`, `.dvc/config` |
| 5.6 | `eval_local._align` — заменить `continue` на явную ошибку при отсутствующем предсказании | `m3m6/.../eval_local.py:17-18` |
| 5.7 | Убрать неиспользуемый `seed` из группового сплиттера или начать использовать | `splits.py` |
| 5.8 | Пометить `configs/few_shot.yaml` как неактуальный; вернуть 7 реальных примеров из `m3m6/configs/few_shot.yaml` | `configs/few_shot.yaml` |

Рекомендация по 5.5: **удалить DVC**. Он ссылается на несуществующие пути, remote `/tmp/dvc-rag-m3m6` не существует, реальный alfa-пайплайн под ним никогда не был заведён. Поддерживать иллюзию воспроизводимости хуже, чем её отсутствие; `run.yaml` + закоммиченные `scores.jsonl` покрывают потребность.

---

## 6. Задача 0.6 — перепрогон существующих методов

Порядок по возрастанию стоимости:

| Метод | Что нужно | Стоимость |
|---|---|---|
| majority, keyword | ничего | секунды |
| surface, surface_e5 | переобучить OOF по фолдам (e5-косинусы уже в кэше) | минуты |
| independent (rule-based) | прогон по 2233 | минуты |
| lettucedetect | прогон по 2233, GPU желателен | ~20 мин |
| encoder | OOF-обучение: 25 моделей (5×5) | см. фазу 1, обсуждается |
| m3 zero_shot / few_shot | прогон по 2233 через vLLM | ~15 мин на A100 |
| m3 GEPA-варианты | прогон по 2233 | ~15 мин |
| m6 | не перепрогонять — переопределяется в фазе 3 | — |

**Про энкодер и OOF.** Полный 5×5 OOF для энкодера — это 25 обучений. При ~2 ч на обучение это 50 GPU-ч, неприемлемо. Компромисс: **5-fold без повторов** (5 обучений, ~10 GPU-ч) для OOF-скоров, а разброс оценивать бутстрэпом по кейсам, а не по повторам. Это фиксируется в `report.json` как `n_repeats: 1` и явно оговаривается при сравнении.

---

## 7. Критерии приёмки фазы

- [ ] `data/splits/folds.json` закоммичен, `prepare_splits.py --check` зелёный, проверки в CI
- [ ] утечка по нормализованному вопросу между фолдами = 0%, near-duplicate < 2%
- [ ] base rate по фолдам в пределах ±2 п.п. от 72.25%
- [ ] `Prediction.scores` в схеме; старые артефакты мигрированы
- [ ] `evaluation/{protocol,bootstrap,nullcal}.py` покрыты тестами
- [ ] `evaluate_cv.py` воспроизводит 10 опубликованных чисел на старом протоколе (1e-6)
- [ ] `report.json` без ДИ и без `null_percentile` не собирается
- [ ] минимум 6 методов перепрогнаны в формате `scores.jsonl` на 2233 кейса
- [ ] `tests/test_splits_leakage.py` фиксирует разрыв 1-NN между протоколами
- [ ] `README.md` и `docs/experiments.md` не содержат сравнений между протоколами

---

## 8. Ожидаемый эффект

Метрика не вырастет. Изменится следующее:

| | До | После |
|---|---|---|
| Ширина 95% ДИ | ±0.070 (n=223) | ±0.022 (n≈2100) |
| sd между запусками | 0.032 (один сплит) | ~0.007 (5 повторов) |
| Оптимизм порога | +0.035…+0.053 | ~0 (OOF) |
| Число протоколов в кодовой базе | 4 | 1 |
| Различимая разница между методами | > 0.14 | > 0.045 |

Последняя строка — главная. Сейчас статистически различить методы невозможно в принципе; после фазы 0 различимы разницы порядка 0.045, а это уже меньше, чем эффект стэкинга (+0.029… на границе) и заведомо меньше, чем ожидаемый эффект от энкодера на 8192.
