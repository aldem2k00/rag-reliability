# Ноутбуки — контур запуска на Yandex DataSphere

> Ищете разбор метода, а не пусковую установку? Он рядом:
> [`notebooks/standalone/`](standalone/README.md) — по одному самостоятельному
> ноутбуку на методы 3 и 6, с гипотезами, кодом метода в ячейках и метриками.
> Этот каталог — про другое: как поднять окружение и запустить корпусный прогон.

Ноутбуки `00`–`40` и ничего больше. Каждый — **пусковая установка**: поднимает
окружение, проверяет железо, поднимает vLLM, вызывает CLI репозитория. Ни одной
строки бизнес-логики в ячейках.

Причина жёсткая, а не стилистическая. Единственное, что в проекте работает
безупречно, — `run.yaml` с git-хэшем рядом с каждым прогоном: 10 из 10 метрик
воспроизвелись до седьмого знака. Расчёт, живущий в ячейке, не попадает ни в
`run.yaml`, ни в git-хэш, и воспроизводимость умирает первой.

**Не хватает чего-то в CLI — это баг CLI.** Список известных пробелов и требуемых
контрактов — в [`docs/datasphere.md`](../docs/datasphere.md) §8.

Правило закреплено тестами: `tests/test_notebooks.py` разбирает `.ipynb` без
исполнения и проверяет, что ячейки не определяют функций, не импортируют счётные
библиотеки, клонируют `integration` и не вызывают `split_samples`.

---

## Порядок

| Ноутбук | Что делает | Конфигурация | Время |
|---|---|---|---:|
| [`00_setup.ipynb`](00_setup.ipynb) | стек, клон `integration`, железо, vLLM, **смоук на logprobs** | `g2.1` | ~15 мин |
| [`10_score_judge.ipynb`](10_score_judge.ipynb) | прогоны судьи по всему корпусу | `g2.1` | 15–50 мин на вариант |
| [`20_train_encoder.ipynb`](20_train_encoder.ipynb) | OOF-обучение энкодера по фолдам | `g2.1` | 3 мин – 1 ч |
| [`30_finetune_judge.ipynb`](30_finetune_judge.ipynb) | fine-tuning судьи Qwen2.5-7B по фолдам | `g2.1` | ~1.5 ч на фолд |

`00_setup.ipynb` запускается первым в каждой GPU-сессии: остальные три
предполагают уже поднятый vLLM и склонированный репозиторий.

Всё, что дольше двух часов, здесь не запускается — VM ноутбука останавливается при
простое. Для таких прогонов готовы конфиги [`jobs/*.yaml`](../jobs):
GEPA (~7 ч на H5), OOF энкодера на 8192 токенах (~10 ч), FT судьи по пяти фолдам
(~7.5 ч).

---

## Обязательный смоук на logprobs

`00_setup.ipynb` заканчивается прогоном на 5 кейсах с двумя ассертами:
`prob_method == "logprobs"` у всех строк и разные значения `m3.p_faith`.

Пропускать нельзя. Токенизатор vLLM отличается от того, что был у OpenRouter, а
извлечение вердикта чувствительно к разбиению `PASS`/`FAIL` на подтокены: при
односимвольном первом подтокене вероятность **молча** становится 0.5 для всех
кейсов. Прогон на 2233 кейса при этом выглядит успешным, а сигнала в нём нет.

---

## Сплит

Разбиение читается из `data/splits/folds_alfa.json`. `split_samples` не вызывается
ни в одном ноутбуке: он даёт стратифицированный, а не group-aware сплит, 24.9%
тестовых строк делят вопрос с train, и полученные им числа несравнимы с числами на
group-сплитах и завышены.

---

## Full-FT ноутбуки Methods 1/2

Раньше `qwen7b_full_finetune*.ipynb` были вычищены из launcher-контура: они
клонировали ветку `qwen7b-notebook`, резали корпус через `split_samples` и
держали обучение с оценкой в ячейках. Их роль для FT судьи по фолдам перешла к
`30_finetune_judge.ipynb`.

В `main` полный FT Methods 1/2 снова задокументирован отдельными ноутбуками —
они лежат рядом; см. секцию ниже.

---

## Ссылки

- [`docs/datasphere.md`](../docs/datasphere.md) — операционная памятка, чеклист,
  типовые проблемы, пробелы в CLI.
- [`docs/specs/90_DATASPHERE_runbook.md`](../docs/specs/90_DATASPHERE_runbook.md) —
  полный runbook: конфигурации, раскладка хранилища, бюджеты прогонов.

---

## Full fine-tune notebooks (Methods 1/2)

Из `main` снова лежат рядом с launcher-контуром:

| Notebook | Role |
|---|---|
| [`qwen7b_full_finetune.ipynb`](qwen7b_full_finetune.ipynb) | Full FT Qwen2.5-7B (direct/marker) for Colab/Kaggle/DataSphere |
| [`qwen7b_full_finetune_datasphere.ipynb`](qwen7b_full_finetune_datasphere.ipynb) | DataSphere-oriented variant of the same full-FT path |
| [`eval_finetuned_datasphere.ipynb`](eval_finetuned_datasphere.ipynb) | Evaluate a saved full-FT checkpoint on DataSphere |

Для корпусного FT судьи по фолдам в этом репозитории основной путь — всё ещё
[`30_finetune_judge.ipynb`](30_finetune_judge.ipynb) + `jobs/ft_judge_fold*.yaml`.
Подробности по full-FT ноутбукам Methods 1/2 — в
[`docs/qwen7b_full_ft_results.md`](../docs/qwen7b_full_ft_results.md).

