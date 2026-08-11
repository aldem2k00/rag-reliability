# 90 — Runbook: выполнение GPU-этапов в Yandex DataSphere

Операционный документ. Читается перед первым запуском на GPU и держится открытым во время работы.

---

## 1. Выбор конфигурации

| Конфигурация | vCPU | GPU | VRAM | RAM | Для чего |
|---|---:|---|---:|---:|---|
| `c1.8` | 8 | — | — | 64 GB | фаза 0 целиком, стэкинг, coverage |
| `gt4i.1` | 8 | 1× T4i | 24 GB | 32 GB | NLI-grounding, LettuceDetect |
| `g1.1` | 8 | 1× V100 | 32 GB | 48–96 GB | энкодер до 4096 токенов |
| **`g2.1`** | 28 | 1× A100 | **80 GB** | 119 GB | **судья через vLLM, энкодер 8192, FT Qwen-7B** |
| `g2.2` | 56 | 2× A100 | 160 GB | 238 GB | только если 7B FT не влезает (не должно) |

**Правило:** всё, что связано с LLM-инференсом или обучением 7B, — `g2.1`. Всё остальное — берите меньше, это в разы дешевле.

Оговорка по V100: это Volta, bf16 нет, flash-attention 2 требует Ampere. Для vLLM и ModernBERT на 8192 берите `g2.1`.

---

## 2. Раскладка хранилища

| Путь | Что | Переживает рестарт VM |
|---|---|---|
| `/home/jupyter/datasphere/project` | проектное хранилище, синхронизируется | да, но ограничено по объёму |
| `/home/jupyter/filestore/<name>` | File Storage, монтируется после активации | да |
| `/tmp`, `~` вне указанных | скрэтч | **нет** |

**Обязательное правило:** кэши судьи, чекпоинты и `scores.jsonl` — на File Storage. Ваш поэлементный кэш делает прогоны резюмируемыми, но только если он переживает рестарт.

```python
BASE = "/home/jupyter/filestore/neurodrive"   # активировать: Ресурсы проекта → Файловое хранилище
REPO = f"{BASE}/rag-reliability"
CACHE = f"{BASE}/cache/m3_judge"
```

---

## 3. Ноутбук как пусковая установка

Сильнейшая сторона проекта — `run.yaml` с git-хэшем рядом с каждым прогоном и артефакты, воспроизводимые до седьмого знака. Если логика экспериментов переедет в ячейки, это умрёт первым.

**Целевой ноутбук — четыре ячейки, вся логика в репозитории.**

### Ячейка 1 — окружение

```python
BASE   = "/home/jupyter/filestore/neurodrive"
REPO   = f"{BASE}/rag-reliability"
BRANCH = "feature/m3-m6-thresholds"       # НЕ qwen7b-notebook — та ветка устарела

import os, sys, subprocess
if not os.path.isdir(REPO):
    subprocess.check_call(["git","clone","-b",BRANCH,
                           "https://github.com/aldem2k00/rag-reliability.git", REPO])
else:
    subprocess.check_call(["git","-C",REPO,"pull","--ff-only"])
subprocess.check_call([sys.executable,"-m","pip","install","-q","-e",f"{REPO}[m6]"])
sys.path.insert(0, f"{REPO}/src")
print(subprocess.check_output(["git","-C",REPO,"rev-parse","--short","HEAD"]).decode())
```

### Ячейка 2 — проверка железа

```python
import torch, psutil
n = torch.cuda.device_count()
vram = torch.cuda.get_device_properties(0).total_memory/1e9 if n else 0
print(f"GPU {torch.cuda.get_device_name(0) if n else '—'} | VRAM {vram:.0f}GB | RAM {psutil.virtual_memory().total/1e9:.0f}GB")
assert n >= 1, "выбрана CPU-конфигурация"
```

### Ячейка 3 — vLLM в фоне (нужен для M3 и для DSPy/GEPA)

```python
import subprocess, time, requests, os
os.makedirs(f"{BASE}/logs", exist_ok=True)
subprocess.Popen(
    f"vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 "
    f"--max-model-len 8192 --gpu-memory-utilization 0.85 --max-logprobs 25 "
    f"> {BASE}/logs/vllm.log 2>&1", shell=True)

for _ in range(120):                       # ждём подъёма, до 10 минут
    try:
        if requests.get("http://localhost:8000/v1/models", timeout=2).ok:
            print("vLLM up"); break
    except Exception: pass
    time.sleep(5)
else:
    raise RuntimeError(f"vLLM не поднялся, смотри {BASE}/logs/vllm.log")
```

`--max-logprobs 25` обязателен: извлечение вероятностей запрашивает `top_logprobs=20`.

### Ячейка 4 — запуск через CLI, `run.yaml` пишется как обычно

```python
!cd {REPO} && OPENAI_API_KEY=dummy python scripts/score.py \
    --method m3_judge --variant zero_shot \
    --data data/organizers.jsonl \
    --m3-api-base http://localhost:8000/v1 \
    --m3-cache-dir {CACHE} --m3-concurrency 16 \
    --output predictions/m3_judge/zero_shot/scores.jsonl
```

---

## 4. Обязательный смоук перед каждым полным прогоном судьи

**Не пропускать.** Токенизатор vLLM отличается от того, что был у OpenRouter-провайдера, а `_verdict_positions` (`logprobs.py:19-46`) чувствителен к разбиению `PASS`/`FAIL` на подтокены. При односимвольном первом подтокене `_pass_prob` молча возвращает 0.5 для всех кейсов.

```python
!cd {REPO} && python scripts/score.py --method m3_judge --variant zero_shot --limit 5 \
    --m3-api-base http://localhost:8000/v1 --m3-cache-dir /tmp/smoke --output /tmp/smoke.jsonl

import json
rows = [json.loads(l) for l in open("/tmp/smoke.jsonl")]
methods = [r.get("prob_method") for r in rows]
print(methods)
assert all(m == "logprobs" for m in methods), (
    "PASS/FAIL режется на подтокены — чинить _verdict_positions ДО полного прогона "
    "(см. 20_PHASE1, задача 1.2)")
probs = [r["scores"]["m3.p_faith"] for r in rows]
assert len(set(round(p, 3) for p in probs)) > 1, "все вероятности одинаковы — извлечение сломано"
```

---

## 5. Бюджеты прогонов

Оценки для Qwen2.5-7B на одной A100 80GB через vLLM, корпус 2233 кейса.

| Прогон | Prompt-токенов | Output-токенов | Время |
|---|---:|---:|---:|
| Судья, холистический, 1 сэмпл | 6.7 M | 0.7 M | **~15 мин** |
| Судья, ось relevance отдельно (без чанков) | 1.3 M | 0.4 M | ~5 мин |
| Судья, self-consistency k=8 | 6.7 M (prefix cache) | 5.4 M | ~50 мин |
| Судья, per-chunk ×8 | ~7 M | 2.2 M | ~40 мин |
| GEPA, один прогон (~2000 rollouts) | ~6 M | ~1.2 M | ~1.2 ч |
| GEPA, H5 полностью (3 сида × 2 варианта) | — | — | **~7 ч** |
| Энкодер 8192, одно обучение (3 эпохи, 1787 кейсов) | — | — | ~2 ч |
| Энкодер, 5-fold OOF | — | — | **~10 ч** |
| FT Qwen-7B, один фолд | — | — | ~1.5 ч |
| NLI-grounding, весь корпус (~90k пар) | — | — | **~10 мин** |

**Вывод, который меняет планирование:** локальный инференс почти бесплатен по сравнению с этапом на OpenRouter. Дорогие позиции — GEPA (7 ч) и OOF-обучения (10 ч), а не прогоны судьи.

---

## 6. Долгие прогоны: DataSphere Jobs

VM ноутбука останавливается при простое; прогон на 6–8 часов (GEPA, OOF-обучения) рискует умереть на середине.

**Правило: всё, что дольше 2 часов, — через Jobs, а не через ноутбук.**

```bash
pip install datasphere
```

`config.yaml`:

```yaml
name: gepa-h5-markers-seed0
desc: GEPA prompt evolution, markers variant, seed 0
cmd: >
  python scripts/run_gepa.py
  --train-data data/splits/gepa_train.jsonl
  --val-data data/splits/gepa_pareto.jsonl
  --variant markers --seed 0 --auto medium
  --model Qwen/Qwen2.5-7B-Instruct --api-base http://localhost:8000/v1
env:
  python:
    version: "3.11"
    pip-file: requirements.txt
inputs:
  - data/
  - configs/
outputs:
  - artifacts/gepa/markers_seed0/
cloud-instance-type: g2.1
```

```bash
datasphere project job execute -p <project-id> -c config.yaml
```

Задание исполняется на VM независимо от JupyterLab: сессия может закрыться, прогон продолжится.

---

## 7. Защита от потери прогона

| Мера | Как |
|---|---|
| Кэш судьи на File Storage | `--m3-cache-dir {BASE}/cache/m3_judge` — прерванный прогон продолжается с места остановки |
| Чекпоинты обучения | `SAVE_STRATEGY="epoch"` вместо `"no"` (в текущем FT-ноутбуке стоит `"no"` — восьмичасовой прогон теряется целиком) |
| Промежуточные скоры | писать `scores.jsonl` инкрементально, а не в конце |
| Логи vLLM | `{BASE}/logs/vllm.log`, не `/tmp` |
| Артефакты в git | после каждого завершённого прогона: `git add predictions/... && git commit` |

---

## 8. Чеклист перед запуском GPU-этапа

- [ ] Фаза 0 закрыта: `data/splits/folds.json` закоммичен, `prepare_splits.py --check` зелёный
- [ ] Ноутбук клонирует **рабочую** ветку, а не `qwen7b-notebook`
- [ ] Обучение/инференс читает `folds.json` и **не вызывает** `split_samples`
- [ ] Кэш и чекпоинты — на File Storage
- [ ] `logprobs.py` починен (задача 1.2), кэш инвалидирован
- [ ] Смоук на 5 кейсах даёт 100% `prob_method == "logprobs"` и разные вероятности
- [ ] Для прогона длиннее 2 ч подготовлен `config.yaml` для Jobs
- [ ] `run.yaml` пишет `dirty` флаг; рабочее дерево чистое перед запуском

---

## 9. Типовые проблемы

| Симптом | Причина | Что делать |
|---|---|---|
| `AssertionError` на VRAM в FT-ноутбуке | выбрана не `g2.1` | сменить конфигурацию, перезапустить VM |
| vLLM не поднимается, лог пуст | не хватило VRAM под KV-кэш | понизить `--gpu-memory-utilization` до 0.75, `--max-model-len` до 4096 |
| Все `p_faith == 0.5` | `PASS` режется на подтокены | задача 1.2(а,б); не запускать полный прогон |
| `prob_method == "regex"` на всех строках | провайдер/сервер не отдаёт logprobs | проверить `--max-logprobs` у vLLM и `logprobs=True` в запросе |
| Прогон завис после N кейсов | оборвалась сессия | перезапустить ту же команду — кэш подхватит сделанное |
| `File Storage` не смонтирован | не перезапущена VM после активации | Ресурсы проекта → Файловое хранилище → активировать → **рестарт VM** |
| Модель схлопнулась в (1,1) | дисбаланс классов | взвешенный лосс + oversampling; контроль `degenerate.const_share` после каждой эпохи |
| HF-загрузка падает | нет токена для gated-моделей | `HF_TOKEN` в переменных окружения проекта |

---

## 10. Что делать на CPU, пока GPU не нужен

Фаза 0 целиком и задача 1.1 (стэкинг) не требуют GPU. Логично закрыть их первыми: тогда к моменту подъёма A100 будут готовы канонический сплит, CV-контур и стэк-бейзлайн, относительно которого честно меряется всё остальное.

Обратный порядок означает, что GPU-часы уйдут на прогоны, которые потом придётся пересчитывать на другом сплите.

---

## Источники

- [Yandex DataSphere — конфигурации вычислительных ресурсов](https://yandex.cloud/en/docs/datasphere/concepts/configurations)
- [Yandex DataSphere — Jobs](https://yandex.cloud/en/docs/datasphere/concepts/jobs)
- [Yandex DataSphere — предустановленное ПО](https://cloud.yandex.com/en/docs/datasphere/concepts/preinstalled-packages)
