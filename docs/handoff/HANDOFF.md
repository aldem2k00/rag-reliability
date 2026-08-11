# HANDOFF — реализация роадмапа RAG-reliability силами агентов

Точка входа. Читается первым — человеком и каждым агентом.

**Репозиторий:** `MurkaSelebry/rag-reliability` (форк).
**База всех веток:** `upstream/feature/m3-m6-thresholds`.
**Спецификации:** `docs/specs/00_ARCHITECTURE.md` … `90_DATASPHERE_runbook.md`.
**Карточки задач:** `docs/handoff/tasks/<ID>.md` — одна задача = один агент = одна ветка = один PR.

---

## 1. Ответ на главный вопрос: сколько кода можно написать до DataSphere

**Примерно 90%. Пишите всё, кроме фазы 4.**

Причина: в репозитории уже есть dummy-бэкенды (`--m3-backend dummy`, `m6/dummy.py`, `dummy_model.py`) и ~150 тестов. Любой инференс-код тестируется без GPU. GPU нужен только для *запусков*, не для *написания*.

| Что | Пишется без GPU | Тестируется без GPU | Проверяется на реальных данных без GPU |
|---|---|---|---|
| Фаза 0 целиком | да | да | **да, полностью** — это чистый CPU на 2233 кейсах |
| 1.1 Стэкинг | да | да | **да** — артефакты предсказаний уже в репозитории |
| 1.2 Починка logprobs | да | да | да — пересчёт по закоммиченным `meta.raw` |
| 1.3 Энкодер | да | да (мок-модель) | нет — нужен GPU для обучения |
| 1.4 Прогон судьи | да | да (dummy-бэкенд) | нет — нужен vLLM |
| 2.x Метод 3 | да | да (dummy-бэкенд) | частично — промпты можно проверить на 20 кейсах через OpenRouter |
| 3.x Метод 6 grounding | да | да (мок NLI) | частично — mDeBERTa на CPU идёт медленно, но 50 кейсов реально |
| 4.x Исследовательские | **нет** | — | — |

**Почему фазу 4 писать нельзя.** Её выбор зависит от чисел фаз 1–3. Оставить ли M6 вообще — решается по ROC-AUC из задачи `C4`. Нужна ли span-голова — зависит от того, упёрся ли энкодер. Писать этот код сейчас значит с высокой вероятностью его выбросить.

**Стратегия:** волны 1–4 ниже пишутся и тестируются локально до единого GPU-часа. DataSphere превращается в этап «запустить готовое и собрать артефакты». Это же снимает главный операционный риск — отладка в ноутбуке с умирающей сессией.

---

## 2. Что нужно знать про проект за две минуты

Автоматический quality gate для русскоязычного банковского RAG-бота. Вход — `(диалог + вопрос, до 8 чанков, ответ бота)`, выход — `faithfulness`, `relevance`, `reliable = faith ∧ rel`. Метрика — macro-F1 по `reliable`. Только локальные модели.

**Текущее состояние — то, что определяет всю работу:**

- корпус 2233 уникальных кейса, 72.25% reliable;
- лучший результат 0.5982 (surface-логрег), судья 0.5841, энкодер 0.5879;
- **эти числа статистически неразличимы**: n_test = 223, ширина 95% ДИ ±0.070, p = 0.42 при сравнении лучших двух;
- **все val-числа Метода 3 лежат внутри шума процедуры подбора порогов** (p95 шума = 0.5947);
- на текущей ветке сплит **протекает**: 24.9% тестовых строк делят вопрос с train, 1-NN-запоминатель даёт 0.6278 — выше всех опубликованных чисел.

Отсюда порядок: сначала измерительный контур, потом методы. Фаза 0 блокирует всё.

---

## 3. Подготовка репозитория (делает человек, один раз)

```bash
cd rag-reliability-selebry
git remote add upstream https://github.com/aldem2k00/rag-reliability.git
git fetch upstream
git checkout -b integration upstream/feature/m3-m6-thresholds
git push -u origin integration

# положить спецификации и handoff в репозиторий
mkdir -p docs/specs docs/handoff
cp ../RAG_reliability_specs/*.md docs/specs/
cp -r <распакованный handoff>/* docs/handoff/
cp <распакованный handoff>/AGENTS.md .          # Claude Code читает его из корня
git add -A && git commit -m "docs: спецификации и handoff" && git push
```

`integration` — целевая ветка всех PR. В `main` ничего не льём до конца фазы 1.

---

## 4. Как запускать агентов параллельно

Каждая задача — отдельный git worktree, чтобы агенты не мешали друг другу:

```bash
git worktree add ../wt-A1 -b task/A1-splits integration
git worktree add ../wt-A2 -b task/A2-stats  integration
git worktree add ../wt-A3 -b task/A3-schema integration
# ...
```

Затем в каждом worktree запускается свой агент. Промпт агента — ровно один:

```
Прочитай docs/handoff/AGENTS.md и docs/handoff/tasks/A1.md и выполни задачу A1
целиком. Ничего за пределами своего списка владения не меняй.
Когда закончишь — прогони критерии приёмки из карточки и открой PR в ветку integration.
```

Больше ничего в промпт класть не нужно: карточка самодостаточна.

**Слияние:** PR вливаются в `integration` в порядке волн. После каждой волны — `make check` на `integration` и обновление worktree следующей волны через `git rebase integration`.

---

## 5. Волны и параллелизм

```
ВОЛНА 1 — 5 агентов параллельно, пересечений по файлам нет
  A1 splits          A2 stats        A3 schema       A4 logprobs     A5 hygiene
       └──────────────────┴───────────────┘                │               │
                          ▼                                │               │
ВОЛНА 2 — 2 агента                                         │               │
  B1 protocol+evaluate_cv          B2 score-cli            │               │
       └────────────────┬───────────────┘                  │               │
                        ▼                                  ▼               ▼
ВОЛНА 3 — 4 агента параллельно
  C1 stacking      C2 encoder      C3 m3-axes      C4 m6-grounding
       └────────────────┴──────────────┬──────────────────┘
                                       ▼
ВОЛНА 4 — 4 агента параллельно
  D1 m3-perchunk   D2 gepa   D3 ft-notebook   D4 reporting
```

| Волна | Задачи | Параллельно | Дней при параллельной работе |
|---|---|---:|---:|
| 1 | A1–A5 | 5 | 2 |
| 2 | B1, B2 | 2 | 1.5 |
| 3 | C1–C4 | 4 | 3 |
| 4 | D1–D4 | 4 | 3 |

Итого ~9–10 дней вместо ~24 последовательных.

---

## 6. Матрица владения файлами

**Железное правило: агент меняет только файлы из своей колонки «владеет». Всё остальное — read-only.** Пересечений в одной волне нет — это и делает параллелизм безопасным.

| ID | Ветка | Владеет (exclusive) | Читает |
|---|---|---|---|
| **A1** | `task/A1-splits` | `src/rag_reliability/splits.py`, `scripts/prepare_splits.py`, `data/splits/*`, `tests/test_splits*.py` | `dataset.py`, `schema.py`, спека `10_PHASE0` §1 |
| **A2** | `task/A2-stats` | `src/rag_reliability/evaluation/{__init__,bootstrap,nullcal}.py`, `src/rag_reliability/metrics.py`, `tests/test_bootstrap.py`, `tests/test_nullcal.py`, `tests/test_metrics.py` | `schema.py`, спека `10_PHASE0` §3 |
| **A3** | `task/A3-schema` | `src/rag_reliability/schema.py`, `scripts/migrate_predictions.py`, `tests/test_schema.py`, `tests/test_migrate.py` | спека `00_ARCHITECTURE` §2.3 |
| **A4** | `task/A4-logprobs` | `src/rag_reliability/methods/m3/{logprobs,parsing,judge_client}.py`, `tests/test_m3_logprobs.py`, `tests/test_m3_judge_client.py`, `tests/test_parsing.py` | спека `20_PHASE1` §2 |
| **A5** | `task/A5-hygiene` | `src/rag_reliability/run_meta.py`, `predictions/**` (перенос), `README.md`, `dvc.*`, `.dvc/`, `configs/few_shot.yaml`, `Makefile` | спека `10_PHASE0` §5 |
| **B1** | `task/B1-protocol` | `src/rag_reliability/evaluation/protocol.py`, `scripts/evaluate_cv.py`, `tests/test_protocol*.py` | A1, A2, A3 |
| **B2** | `task/B2-score-cli` | `scripts/score.py`, `src/rag_reliability/methods/registry.py`, `tests/test_score_cli.py`, `tests/test_registry.py` | A3 |
| **C1** | `task/C1-stacking` | `src/rag_reliability/stacking/*`, `scripts/run_stack.py`, `tests/test_stacking.py` | B1 |
| **C2** | `task/C2-encoder` | `src/rag_reliability/methods/encoder/*`, `scripts/train_encoder_baseline.py`, `tests/test_encoder*.py` | B1, B2 |
| **C3** | `task/C3-m3-axes` | `src/rag_reliability/methods/m3/{prompts,axes,selfconsistency}.py`, `configs/prompts/*`, `scripts/run_m3.py`, `tests/test_m3_*.py` (кроме занятых A4) | A4, B2 |
| **C4** | `task/C4-m6-grounding` | `src/rag_reliability/methods/m6/{grounding,coverage,nli,features}.py`, `scripts/score_m6_grounding.py`, `tests/test_m6_*.py` | B2 |
| **D1** | `task/D1-perchunk` | `src/rag_reliability/methods/m3/perchunk.py`, `tests/test_m3_perchunk.py` | C3 |
| **D2** | `task/D2-gepa` | `src/rag_reliability/methods/m3/gepa*.py`, `scripts/run_gepa.py`, `tests/test_gepa*.py` | C3 |
| **D3** | `task/D3-notebooks` | `notebooks/*`, `docs/datasphere.md` | B1, C2, спека `90_DATASPHERE` |
| **D4** | `task/D4-reporting` | `docs/experiments.md`, `scripts/make_leaderboard.py`, `docs/report/*` | всё |

**Конфликтные точки, о которых надо помнить:**

- `schema.py` трогает только A3. Все остальные ждут его мержа — но не блокируются: пишут код против контракта, описанного в `00_ARCHITECTURE.md` §2.3, и делают `git rebase integration` после мержа A3.
- `metrics.py` — только A2. C-агенты не добавляют туда метрики, а кладут свои в `evaluation/`.
- `registry.py` — только B2.
- `tests/test_m3_*.py` разделены между A4 и C3 поимённо: смотрите таблицу.

---

## 7. Контракты, которые фиксируются до начала работ

Агенты пишут против этих сигнатур, не согласовывая их между собой.

### 7.1 `Prediction.scores`

```python
class Prediction(BaseModel):
    ...
    scores: dict[str, float] = Field(default_factory=dict)   # ключ: "<метод>.<сигнал>"
```

Префиксы закреплены за методами: `surf.`, `m3.`, `m6.`, `enc.`, `ld.`, `stack.`

### 7.2 `data/splits/folds.json`

```json
{"schema_version": 1,
 "corpus": {"path": "...", "sha256": "...", "n": 2233},
 "config": {"n_folds": 5, "n_repeats": 5, "seed": 2233, "near_dup_threshold": 0.90,
            "use_chunk_key": true},
 "stats": {"n_groups": 0, "largest_group": 0, "oversized_groups": [],
           "excluded_ids": 0, "pos_rate_global": 0.0, "pos_rate_by_fold": [[]],
           "leak_check": {"query_overlap": 0.0, "near_dup_0.9": 0.0, "chunk1_overlap": 0.0}},
 "assignment": {"alfa_xxxxxxxxxxxx": [0, 3, 1, 4, 2]}}
```

`assignment[id][r]` — номер фолда кейса в повторе `r`. Исключённые кейсы (oversized-группы) отсутствуют в `assignment`.

### 7.3 `evaluation/protocol.py`

```python
def evaluate_cv(samples, predictions, folds, *, score_fn, fit_fn=None,
                grid_step=0.01) -> CVResult: ...
```

### 7.4 Артефакт метода

```
predictions/<method>/<variant>/
    scores.jsonl     ровно 2233 строки (или n_evaluated, если метод частичный)
    report.json      схема EvaluationReport, обязательные ci95 и null_percentile
    run.yaml         конфиг + git hash + dirty + seed
```

### 7.5 CLI

```bash
python scripts/score.py --method <name> --variant <name> --data <jsonl> --output <scores.jsonl> [--limit N]
python scripts/evaluate_cv.py --data <jsonl> --folds data/splits/folds.json \
       --scores <scores.jsonl> --score-expr "<выражение по ключам scores>" --output <report.json>
```

---

## 8. Правила, общие для всех агентов

Продублированы в `AGENTS.md`, который Claude Code читает автоматически.

**П1. Ни одного `dict.get(key, default)` для фич и скоров.** Отсутствие ключа — исключение. В M6 молчаливый дефолт (`predict.py:41-43`) привёл к тому, что битая строка фич трактовалась как идеально надёжный кейс.

**П2. Порог никогда не подбирается на данных, на которых отчитываются.** Только внутри train-части фолда.

**П3. Ни одного числа без интервала.** `report.json` без `ci95` и `null_percentile` не собирается — это валидируется схемой.

**П4. Не вызывать `split_samples` в новом коде.** Разбиение читается из `folds.json`. Единственное исключение — регресс-тест, воспроизводящий старые числа.

**П5. Тесты пишутся вместе с кодом, а не после.** Каждая карточка содержит минимальный набор; агент вправе добавить больше.

**П6. Не менять сигнатуры из §7.** Если сигнатура кажется неверной — остановиться и написать об этом в PR, а не менять в одностороннем порядке.

**П7. Не трогать чужие файлы.** Нужна правка в чужом файле — описать её в разделе «Требуется от других» в PR.

**П8. Русские комментарии и докстринги допустимы** (в репозитории уже смешанный стиль), но имена и API — на английском.

**П9. Перед PR:** `make check` (тесты + ruff) зелёный, критерии приёмки из карточки выполнены поимённо.

---

## 9. Чего делать нельзя

| Запрет | Почему |
|---|---|
| Запускать GPU-обучение до закрытия волны 2 | числа придётся пересчитывать на новом сплите |
| Добавлять новые варианты методов в реестр «на будущее» | 15 вариантов при n=223 дали оптимизм отбора +0.049 |
| Оптимизировать метрику GEPA по accuracy | при базовой ставке 72% это уводит судью в режим «всегда PASS» |
| Использовать `predictions/cloud/**` как результаты | это синтетика на 20–30 кейсах; A5 переносит её в `pseudo_debug/` |
| Сравнивать числа, полученные на разных `folds.json` | фиксируйте sha256 фолдов в `report.json` |
| Переиспользовать старый кэш судьи после A4 | кэш хранит выведенные вероятности, а не сырые logprobs; починка будет невидима |

---

## 10. Порядок приёмки

После каждой волны — на ветке `integration`:

```bash
make check                                    # тесты + ruff
python scripts/prepare_splits.py --check      # с волны 1
python scripts/evaluate_cv.py --help          # с волны 2
```

Плюс контрольные точки из спецификаций:

| После волны | Что должно быть верно |
|---|---|
| 1 | `folds.json` закоммичен, утечка по вопросу = 0%, `_pass_prob` монотонен, `predictions/pseudo_debug/` существует |
| 2 | `evaluate_cv.py` воспроизводит 10 старых чисел до 1e-6; `score.py` пишет `scores.jsonl` на весь корпус |
| 3 | стэк ≥ 0.62 macro-F1 на 5×5 CV при σ < 0.02; у M6-grounding есть первый в истории проекта ROC-AUC |
| 4 | ноутбуки DataSphere читают `folds.json`; лидерборд собирается автоматически с ДИ |

---

## 11. Список карточек

| ID | Задача | Волна | Спека |
|---|---|---|---|
| `A1` | Group-aware сплиты и фолды | 1 | `10_PHASE0` §1 |
| `A2` | Статистика: бутстрэп, нулевая калибровка, операционные метрики | 1 | `10_PHASE0` §3 |
| `A3` | Расширение схемы и миграция артефактов | 1 | `00_ARCH` §2.3, `10_PHASE0` §2 |
| `A4` | Починка извлечения вероятностей из logprobs | 1 | `20_PHASE1` §2 |
| `A5` | Гигиена репозитория | 1 | `10_PHASE0` §5 |
| `B1` | Протокол оценки и `evaluate_cv.py` | 2 | `10_PHASE0` §3.1, §4 |
| `B2` | Единый CLI скоринга по всему корпусу | 2 | `00_ARCH` §2.1 |
| `C1` | Стэкинг разнородных сигналов | 3 | `20_PHASE1` §1 |
| `C2` | Энкодер: пакет, 8192 токена, OOF | 3 | `20_PHASE1` §3 |
| `C3` | Метод 3: разделение осей, self-consistency, criteria | 3 | `30_PHASE2` §1–3 |
| `C4` | Метод 6: NLI-grounding и coverage | 3 | `40_PHASE3` §1–3 |
| `D1` | Метод 3: пофрагментная верификация | 4 | `30_PHASE2` §4 |
| `D2` | GEPA заново и протокол H5 | 4 | `30_PHASE2` §5 |
| `D3` | Ноутбуки и Jobs для DataSphere | 4 | `90_DATASPHERE` |
| `D4` | Отчётность и лидерборд | 4 | `10_PHASE0` §4.2 |
