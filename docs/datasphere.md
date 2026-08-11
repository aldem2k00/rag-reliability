# DataSphere — операционная памятка

Короткая шпаргалка на время GPU-сессии. Полный runbook, включая бюджеты токенов и
обоснования — `docs/specs/90_DATASPHERE_runbook.md`; здесь только то, что нужно
держать перед глазами.

---

## 1. Что чем запускать

| Ноутбук / задание | Конфигурация | Время | Через Jobs |
|---|---|---:|:---:|
| `notebooks/00_setup.ipynb` | `g2.1` | ~15 мин (первая установка стека) | нет |
| `notebooks/10_score_judge.ipynb` — zero-shot | `g2.1` | ~15 мин | нет |
| `notebooks/10_score_judge.ipynb` — few-shot | `g2.1` | ~20 мин | нет |
| `notebooks/10_score_judge.ipynb` — разделённые оси | `g2.1` | ~20 мин | нет |
| `notebooks/10_score_judge.ipynb` — self-consistency k=8 | `g2.1` | ~50 мин | нет |
| `notebooks/10_score_judge.ipynb` — пофрагментная | `g2.1` | ~40 мин | нет |
| `notebooks/20_train_encoder.ipynb` — смоук | `g2.1` | ~3 мин | нет |
| `notebooks/20_train_encoder.ipynb` — 2048 токенов, OOF | `g2.1` | ~1 ч | нет |
| `notebooks/30_finetune_judge.ipynb` — смоук | `g2.1` | ~5 мин | нет |
| `notebooks/30_finetune_judge.ipynb` — один фолд | `g2.1` | ~1.5 ч | нет |
| `jobs/gepa_{markers,plain}_seed{0,1,2}.yaml` | `g2.1` | ~1.2 ч каждый, ~7 ч всего | **да** |
| `jobs/encoder_oof.yaml` (8192 токенов) | `g2.1` | ~10 ч | **да** |
| `jobs/ft_judge_fold{0..4}.yaml` | `g2.1` | ~1.5 ч каждый, ~7.5 ч всего | **да** |

**Правило: всё, что дольше двух часов, — через Jobs.** VM ноутбука останавливается
при простое, и шестичасовой прогон умирает на середине.

Меньшие конфигурации дешевле в разы и годятся для части работы: `c1.8` (без GPU) —
фаза 0, стэкинг, coverage; `gt4i.1` (T4i 24 GB) — NLI-grounding и LettuceDetect;
`g1.1` (V100 32 GB) — энкодер до 4096 токенов. Всё, что связано с LLM-инференсом или
обучением 7B, — только `g2.1`: на Volta нет bf16, а flash-attention 2 требует Ampere.

---

## 2. Порядок работы

```
Ресурсы проекта → Файловое хранилище → активировать → РЕСТАРТ VM
Ресурсы проекта → Конфигурация вычислительных ресурсов → g2.1
  ↓
notebooks/00_setup.ipynb   Run All   (клон integration, стек, vLLM, смоук на logprobs)
  ↓
notebooks/10_score_judge.ipynb  /  20_train_encoder.ipynb  /  30_finetune_judge.ipynb
  ↓
длинные прогоны → datasphere project job execute -p <project-id> -c jobs/<config>.yaml
```

Кэш судьи, чекпоинты и логи — на File Storage (`/home/jupyter/filestore/<name>`).
`/tmp` и домашний каталог рестарт VM не переживают.

---

## 3. Ноутбук — пусковая установка, не место для кода

Сильнейшая сторона проекта — `run.yaml` с git-хэшем рядом с каждым прогоном и
артефакты, воспроизводимые до седьмого знака. Если логика эксперимента переедет в
ячейки, это умрёт первым: ячейка не попадает ни в `run.yaml`, ни в git-хэш.

Ноутбук делает ровно четыре вещи: поднимает окружение, проверяет железо, поднимает
vLLM, вызывает CLI. **Не хватает чего-то в CLI — это баг CLI, а не повод писать
расчёт в ячейке.** Правило закреплено тестами `tests/test_notebooks.py`, которые
разбирают `.ipynb` без исполнения.

---

## 4. Чеклист перед запуском GPU-этапа

- [ ] File Storage активировано **и VM перезапущена** (кернел-рестарта недостаточно)
- [ ] выбрана `g2.1`; `00_setup` не упал на ассерте VRAM
- [ ] ноутбук клонирует `integration`, а не `qwen7b-notebook`
- [ ] `python scripts/prepare_splits.py --check` зелёный
- [ ] обучение и инференс читают `folds.json`; `split_samples` не вызывается нигде
- [ ] кэш судьи и чекпоинты указывают на File Storage
- [ ] смоук на 5 кейсах: 100% `prob_method == "logprobs"` и разные вероятности
- [ ] для прогона длиннее 2 ч взят готовый `jobs/*.yaml`
- [ ] рабочее дерево чистое: `run.yaml` пишет флаг `dirty`

---

## 5. Типовые проблемы

| Симптом | Причина | Что делать |
|---|---|---|
| `AssertionError` на VRAM | выбрана не `g2.1` | сменить конфигурацию, **перезапустить VM** |
| File Storage не смонтирован | не перезапущена VM после активации | Ресурсы проекта → Файловое хранилище → активировать → рестарт VM |
| vLLM не поднимается, лог пуст | не хватило VRAM под KV-кэш | `--gpu-memory-utilization 0.75`, `--max-model-len 4096` |
| все `p_faith == 0.5` | `PASS` режется на подтокены | задача A4; полный прогон не запускать, кэш судьи инвалидировать |
| `prob_method == "regex"` везде | сервер не отдаёт logprobs | проверить `--max-logprobs 25` у vLLM |
| прогон завис после N кейсов | оборвалась сессия | повторить ту же команду — кэш подхватит сделанное |
| модель схлопнулась в (1,1) | дисбаланс классов 72/28 | взвешенный лосс + oversampling; смотреть `const_share` после каждой эпохи |
| HF-загрузка падает | нет токена для gated-моделей | `HF_TOKEN` в переменных окружения проекта |
| `ModuleNotFoundError` после установки стека | torch импортировался до `pip install` | Kernel → Restart → Run All |

---

## 6. Защита от потери прогона

| Мера | Как |
|---|---|
| Кэш судьи на File Storage | `--m3-cache-dir {BASE}/cache/m3_judge` — прерванный прогон продолжается |
| Резюме прогона судьи | `scripts/score.py --resume` дочитывает уже посчитанные `id` |
| Чекпоинты обучения | `--save-strategy epoch`, не `no` |
| Инкрементальная запись | `scripts/score.py --flush-every 20` |
| Логи vLLM | `{BASE}/logs/vllm.log`, не `/tmp` |
| Артефакты в git | коммит сразу после каждого завершённого прогона, прямо из ноутбука |
| Длинные прогоны | `jobs/*.yaml` — задание переживает закрытие сессии |

---

## 7. Jobs

```bash
pip install datasphere
datasphere project job execute -p <project-id> -c jobs/gepa_markers_seed0.yaml
```

Конфиг задания состоит из `cmd`, окружения (`jobs/requirements.txt` + `root-paths`),
списков `inputs`/`outputs` и `cloud-instance-type`. Задание исполняется на отдельной
VM: сессия JupyterLab может закрыться, прогон продолжится.

GEPA ходит в `--api-base http://localhost:8000/v1`, а JupyterLab в задании нет, поэтому
vLLM поднимает само задание через `jobs/_with_vllm.sh`. Обучающие задания (энкодер,
FT судьи) сервер не требуют.

---

## 8. Требуется от других задач

Контур ноутбуков написан против CLI. Две точки входа на момент D3 отсутствуют —
это баг CLI, а не повод переносить расчёт в ячейки.

### `scripts/train_ft_judge.py` — нет в репозитории

Нужен `30_finetune_judge.ipynb` и `jobs/ft_judge_fold{0..4}.yaml`. Требуемый контракт:

```bash
python scripts/train_ft_judge.py \
  --data data/alfa.jsonl --folds data/splits/folds_alfa.json --fold 0 --repeat 0 \
  --model Qwen/Qwen2.5-7B-Instruct --mode direct \
  --tuning lora --lora-r 256 --lora-alpha 512 --lora-target-modules all-linear \
  --learning-rate 2e-4 --epochs 3 --max-length 2048 \
  --batch-size 1 --grad-accum 8 \
  --pos-weight-mode balanced --oversample-negatives \
  --save-strategy epoch --save-total-limit 2 --seed 42 --resume \
  --output-dir <checkpoint dir> \
  --predictions-output predictions/alfa/ft_judge/direct_fold0/scores.jsonl \
  --diagnostics-output predictions/alfa/ft_judge/direct_fold0/ft_diagnostics.json \
  [--limit N] [--smoke-only]
```

Обязательное поведение:

- разбиение **только** из `folds.json`; `split_samples` не вызывать;
- `scores.jsonl` общего контракта с `m3.p_faith` / `m3.p_rel`, полученными из logprobs
  токенов вердикта, а не парсингом текста;
- `run.yaml` рядом с артефактом: конфиг + git-хэш + `dirty` + seed;
- `ft_diagnostics.json` с полем `epochs`: `epoch`, `const_share`, `output_entropy`,
  `is_degenerate` после **каждой** эпохи, плюс `collapsed` и `collapse_reason`
  (форма — как у `encoder_diagnostics.json`, `methods/encoder/train.py`);
- взвешенный лосс и oversampling негативов: прогон на 1.5B схлопнулся в константный
  вердикт (1,1) из-за дисбаланса 72/28, и это прошло незамеченным;
- сохранить рабочие части старого ноутбука: ассерт VRAM ≥ 70 GB, 8-bit AdamW,
  gradient checkpointing, bf16, проверка симметрии формата, экспорт на HF Hub.

### `scripts/run_m3.py --prompt-style perchunk` — нет флага

Нужен ячейке «пофрагментная верификация» в `10_score_judge.ipynb`. `perchunk.py`
приходит с задачей D1, но `run_m3.py` в её список владения не входит. Ожидаемые ключи
артефакта: `m3.max_chunk_score`, `m3.mean_chunk_score`, `m3.chunk_disagreement`,
`m3.n_supporting`, `m3.argmax_chunk`; ось — только faithfulness.

### `scripts/score.py --m3-concurrency` — нет флага

`CommandContext.m3_concurrency` в реестре есть, флага в `score.py` нет, а
`build_scorer` для `openai_judge` использует синхронный клиент. Корпусные прогоны
судьи через `score.py` идут последовательно; оценка «~15 мин» из runbook
предполагает параллельность. Пока флага нет, быстрый путь — `scripts/run_m3.py
--concurrency 16`, но он пишет артефакт через `--run-meta`, а не через контракт
`score.py`.

---

## 9. Ссылки

- `docs/specs/90_DATASPHERE_runbook.md` — полный runbook: конфигурации, бюджеты
  токенов и времени, раскладка хранилища.
- `notebooks/README.md` — что делает каждый ноутбук.
- [Yandex DataSphere — конфигурации](https://yandex.cloud/ru/docs/datasphere/concepts/configurations),
  [Jobs](https://yandex.cloud/ru/docs/datasphere/concepts/jobs).
