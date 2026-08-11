<div align="center">

# rag-reliability-judge

**Does a RAG system's answer actually hold up against its context?**
One registry of methods, one `rag-judge` CLI, one `predictions → metrics` contract.

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-passing-2ea44f)
![Lint](https://img.shields.io/badge/lint-ruff-000000)
![Methods](https://img.shields.io/badge/methods-19-8250df)
![Project](https://img.shields.io/badge/SMILES-2026-f5792a)

</div>

---

Part of the team project **"Assessing the Reliability of Responses in RAG
Systems"** @SMILES-2026. Given a `QUESTION`, its `CONTEXT`, and an `ANSWER`,
every method predicts whether the answer is reliable, where

```
reliable = faithfulness AND relevance
```

**faithfulness** = the answer is supported by the context (no hallucination);
**relevance** = the answer actually addresses the question. Nineteen methods —
from a zero-config dummy baseline to LoRA-tuned and Method 3/6 LLM judges —
compete through a single shared contract, so their scores are directly
comparable.

## Reviewing this project

Short on time? Read these three, in this order:

1. **[Results](#results)** below — every headline number with its confidence
   interval and its cohort.
2. **[docs/report/methods_3_6_results.md](docs/report/methods_3_6_results.md)** —
   Methods 3 and 6 in full: what was measured, on which cohort, what each
   hypothesis returned, and the caveats that must travel with each figure.
3. **[notebooks/standalone/](notebooks/standalone/README.md)** — two self-contained
   notebooks that walk through Methods 3 and 6 hypothesis by hypothesis and
   reproduce those numbers.

Verifying without a GPU takes one command — `make install && make check` runs 835
tests and ruff. The notebooks' analysis cells run against the artifacts already
committed under `predictions/`; only the corpus-wide judge runs need an A100.

Three commitments the codebase enforces mechanically, rather than promising in
prose: the split is read from `folds_alfa.json` and never recomputed (an earlier
stratified split leaked 24.9% of questions), the decision threshold is fitted inside
each fold's training part only, and a report without a 95% CI fails schema
validation.

## Contents

- [How it works](#how-it-works)
- [Quickstart](#quickstart)
- [Methods](#methods)
- [The pipeline](#the-pipeline)
- [Results](#results)
- [Metrics](#metrics)
- [Notebooks](#notebooks)
- [Advanced / pipelines](#advanced--pipelines)
- [Status](#status)
- [Documentation map](#documentation-map)
- [Project layout](#project-layout)

## How it works

All methods are registered in one place
([`src/rag_reliability/methods/registry.py`](src/rag_reliability/methods/registry.py))
and driven through a single CLI, `rag-judge`. The registry is the single
source of truth: the CLI, the `run_benchmark` shim, and the Gradio demo all
read from it, so adding a method surfaces it everywhere at once.

![Component architecture](docs/diagrams/architecture.png)

## At a glance

`rag-judge` is the single entry point for running, benchmarking, and scoring
every method against the shared `predictions.jsonl` → `metrics.json`
contract. Nineteen methods are registered, from a zero-config dummy baseline
to LoRA-tuned and Method 3/6 judges; `rag-judge list-methods` prints exactly
what's available and what each one requires. Training and data-prep
pipelines stay as standalone `scripts/*.py` invocations (see
[Advanced / pipelines](#advanced--pipelines)) since they produce artifacts
(adapters, checkpoints, prompts) that methods later consume.

## Quickstart

Requires Python ≥ 3.11. Target hardware: Apple Silicon (MLX); everything
except the `mlx` backend runs anywhere.

```bash
make install        # uv venv + core/dev deps — installs the `rag-judge` console script
make check          # tests + lint
```

<details>
<summary>Optional extras and no-make install</summary>

```bash
make install-mlx                # Apple Silicon: mlx backend + LoRA
make install-lettucedetect      # LettuceDetect feature method
make install-encoder            # RuModernBERT supervised baseline
make install-m6                 # Method 6 NLI/embedding features
make install-cloud              # OpenAI-compatible Method 3 backend
make install-demo               # local Gradio demo UI
make help                       # all shortcuts: dummy, baselines, LoRA, eval
```

Without make:

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
```

</details>

List every registered method, its family, and what it requires:

```bash
rag-judge list-methods
```

Smoke-test the pipeline without any model (dummy backend, no downloads):

```bash
rag-judge run --method dummy_marker --data data/dummy.jsonl --output-dir results/run
```

Run several methods through the shared predictions → metrics contract
(`--methods all` runs every registered method):

```bash
rag-judge benchmark --methods dummy_direct,dummy_marker --data data/dummy.jsonl --output-dir results/benchmark_dummy
```

Each run writes `predictions.jsonl` and `metrics.json` per method plus a
`summary.json` in `--output-dir`.

Real zero-shot baseline (downloads ~840 MB once), then score any predictions
file against gold labels directly:

```bash
rag-judge run --method prompt_direct --data data/dummy.jsonl --output-dir results/run
rag-judge eval --data data/dummy.jsonl --predictions results/run/prompt_direct/predictions.jsonl --output results/run/prompt_direct/metrics.json
```

Launch the local Gradio demo UI:

```bash
make install-demo
rag-judge serve
```

The demo accepts `question`, `context`, `answer`, optional gold labels, and a
method selector sourced from the same registry as the CLI. Methods missing an
artifact or dependency report a clear unavailable status instead of crashing.
It also supports dataset presets, side-by-side method comparison, raw-output
inspection, run history, and batch benchmark command generation.

## Methods

Nineteen methods across nine families. Green nodes run in the Gradio demo
in-process; orange ones are batch-only (need an endpoint, precomputed
features, or an evolved prompt).

![Method taxonomy](docs/diagrams/method-taxonomy.png)

| Method | Family | What it needs | In demo? |
|---|---|---|---|
| `dummy_direct` | dummy | — | yes |
| `dummy_marker` | dummy | — | yes |
| `prompt_direct` | prompt | MLX model | yes |
| `prompt_marker` | prompt | MLX model | yes |
| `lora_direct` | lora | `results/adapters_direct` | yes |
| `lora_marker` | lora | `results/adapters_marker` | yes |
| `lettucedetect` | lettucedetect | `results/lettucedetect/classifier.joblib` | yes |
| `encoder` | encoder | `results/encoder_checkpoints_512_best` | yes |
| `m3_zero_shot` | m3 | MLX model | yes |
| `m3_few_shot` | m3 | `configs/few_shot.yaml` | yes |
| `m3_gepa` | m3 | `configs/m3_gepa_prompt.txt` | batch-only |
| `m3_openai` | m3 | OpenAI-compatible endpoint | batch-only |
| `m3_openai_judge` | m3 | OpenAI-compatible endpoint | batch-only |
| `m3_perchunk` | m3 | OpenAI-compatible endpoint | batch-only |
| `ft_judge` | m3 | `data/splits/folds.json`, GPU ≥ 70 GB | batch-only |
| `m6_selfcheck` | m6 | precomputed m6 features | batch-only |
| `surface` | surface | `data/splits/folds.json` | batch-only |
| `majority` | surface | `data/splits/folds.json` | batch-only |
| `independent` | independent | — | yes |

This table mirrors `registry.METHODS`; run `rag-judge list-methods` for the
same information straight from the code.

<details>
<summary>Method families in one line each</summary>

- **Method 1 — direct** (`mode=direct`): the model outputs
  `{"faithfulness": 0|1, "relevance": 0|1}`.
- **Method 2 — marker** (`mode=marker`): the model names the error type first:
  `{"marker": "...", "faithfulness": 0|1, "relevance": 0|1}`.
- **LettuceDetect features**: token-level scores aggregated into three
  features, then a logistic regression predicts faithfulness and relevance.
- **Method 3 judge** (`m3_*`, `ft_judge`): a prompt judge in zero-shot, few-shot,
  GEPA, split-axis, per-chunk, and OpenAI-endpoint variants; the judge scores
  through the logprobs of its `PASS`/`FAIL` verdict token, so it emits a
  continuous score rather than a bare label. `ft_judge` is the LoRA-tuned
  counterpart used for the fine-tuned-vs-prompt-optimized comparison.
- **Method 6 grounding** (`m6_selfcheck` in the registry): the SelfCheckGPT
  formulation was stopped — the production bot's prompt is unavailable, so a
  proxy generator would measure the gap between two systems rather than the
  bot's hallucination. It was replaced by direct NLI grounding (premise = a
  context chunk, hypothesis = a sentence of the answer), scored by
  `scripts/score_m6_grounding.py`. Twelve features, no LLM calls, ~3 minutes
  over the whole corpus.
- **Supervised encoder**: a RuModernBERT reliability classifier.
- **Surface baselines** (`surface`, `majority`): out-of-fold logistic regression
  over cheap text features, and the base-rate floor every method must clear.
- **Independent rule-based**: heuristic thresholds over faithfulness/relevance
  signals, no model required.

</details>

## The pipeline

`rag-judge benchmark` resolves the requested methods, builds each one's command
from the registry, runs it as a subprocess to produce `predictions.jsonl`, then
scores every method with the same evaluator. Because all families converge on
one `Prediction` schema, the evaluator treats them identically.

![Benchmark pipeline](docs/diagrams/benchmark-pipeline.png)

See [docs/diagrams/](docs/diagrams/README.md) for the source `.puml` files and
a per-sample data-flow diagram.

### The measurement pipeline

`rag-judge benchmark` is the demo path. Every number in [Results](#results) comes
from a second, stricter path — two scripts, one contract:

```bash
# 1. inference: writes scores.jsonl + run.yaml (config, git hash, dirty flag, seed).
#    Resumable; the split is NOT applied here.
python scripts/score.py --method <name> --variant <label> \
  --data data/alfa.jsonl --output predictions/alfa/<method>/<variant>/scores.jsonl

# 2. evaluation: 5x5 CV, threshold fitted inside each fold, bootstrap + null runs
python scripts/evaluate_cv.py --data data/alfa.jsonl --folds data/splits/folds_alfa.json \
  --scores predictions/alfa/<method>/<variant>/scores.jsonl \
  --score-expr "m3.p_faith * m3.p_rel" \
  --output predictions/alfa/<method>/<variant>/report.json
```

The separation is the point: methods emit continuous scores and never binarize
themselves, so the protocol — not the method — owns the threshold, and every method
is judged by identical rules.

## Results

Corpus: 2233 canonical unique cases, 72.4% reliable. Protocol: 5×5 cross-validation
over the group-aware split in `data/splits/folds_alfa.json`, threshold fitted inside
each fold's training part, bootstrap B=10000, and 500 null runs to check a score is
above noise. **A number without a 95% CI is not a result here** — the
`EvaluationReport` schema refuses to serialize one.

| Method | macro-F1 | 95% CI | Above noise | Cohort |
|---|---:|---|---|---:|
| constant "always reliable" (floor) | 0.4203 | — | — | 1480 |
| `independent` (rules, no model) | 0.5122 | [0.4866; 0.5382] | no | 1480 |
| `surface` (text baseline) | 0.5350 | [0.5102; 0.5615] | yes | 1480 |
| **Method 6** — `m6.min_entail` | **0.5339** | [0.5070; 0.5601] | yes | 1480 |
| Method 6 — `rumodernbert` variant | 0.5397 | [0.5128; 0.5658] | yes | 1480 |
| **Method 3** — `surface + m3.p_faith` stacked | **0.6543** | [0.5956; 0.7097] | yes | **331** |

Two things this table does not let you do. You cannot compare the last row with the
others — its cohort is 331 cases (the intersection of source coverages), not 1480.
And you cannot read a 0.02 gap as a difference: the CI width here is roughly ±0.06.

What the last row *does* say: adding `m3.p_faith` to the surface baseline **on the
same cohort** gained **+0.0752**, CI [+0.0158; +0.1344], p = 0.013. That is the only
statistically significant contribution in the project so far, and it came from
Method 3.

Method 6 is a completed, reproducible **negative** result: its best feature reaches
ROC-AUC 0.548 [0.522; 0.575], swapping the NLI model changes nothing, and in the
stack all three of its features score a negative delta. The branch is closed as a
development direction — measured, not abandoned.

Per-cohort numbers, provenance for every figure, and the open caveats are in
**[docs/report/methods_3_6_results.md](docs/report/methods_3_6_results.md)**.

Earlier headline numbers are withdrawn: `0.4194`, `0.4946` and `0.5879` were measured
before the group-aware split (24.9% question leakage) on different protocols and are
not comparable with anything above. See
[the Phase 0 measurement specification](docs/specs/10_PHASE0_измерительный_контур.md).

## Metrics

Reported by `rag-judge eval` (`scripts/evaluate.py`):

- **`reliable_f1_macro`** — primary metric
- `faithfulness_f1_macro`, `relevance_f1_macro`
- `invalid_output_rate` — outputs unparseable even with fallbacks; counted
  conservatively as `faithfulness=0, relevance=0`
- marker mode only: `marker_f1_macro`, `marker_per_class_f1`,
  `marker_confusion` (gold → predicted counts)

## Notebooks

Two contours, deliberately built to opposite rules.

**[`notebooks/standalone/`](notebooks/standalone/README.md)** — one notebook per
method, each readable end to end as an account of that method: the problem, the
hypotheses with their target numbers, the method's own code visible in the cells,
the corpus run, metrics with confidence intervals, and a verdict per hypothesis.
Shaped after `from_organizators/baseline.ipynb`.

| Notebook | Method | Hypotheses | Hardware | Time |
|---|---|---|---|---:|
| `method3_judge.ipynb` | LLM judge (Qwen2.5-7B) | split axes · self-consistency · per-chunk · GEPA · fine-tuning | A100 80 GB | ~2 h + ~15 h in Jobs |
| `method6_grounding.ipynb` | NLI grounding (mDeBERTa) | signal on faithfulness vs relevance · NLI ablation · stack contribution | any GPU, runs on CPU | ~15 min |

**[`notebooks/`](notebooks/README.md)** (`00`–`40`) — launchers for Yandex
DataSphere. Not a single line of business logic in a cell, enforced by
`tests/test_notebooks.py`: a computation living in a cell reaches neither `run.yaml`
nor the git hash, and reproducibility dies first. Runs longer than two hours go
through [`jobs/*.yaml`](jobs) instead, because the notebook VM stops when idle.

## Advanced / pipelines

Training and data-prep steps produce artifacts (adapters, checkpoints,
converted datasets, evolved prompts) that the methods above consume. They are
not part of the `run`/`benchmark`/`eval` contract, so they stay as raw script
invocations.

<details>
<summary>Data prep + supervised encoder baseline</summary>

```bash
python scripts/prepare_data.py \
  --input from_organizators/data/data.zip \
  --output data/organizers.jsonl

python scripts/train_encoder_baseline.py \
  --data data/organizers.jsonl \
  --output results/encoder_baseline_512_best_metrics.json \
  --output-dir results/encoder_checkpoints_512_best \
  --max-length 512 --batch-size 4 \
  --epochs 3 --learning-rate 2e-5 --pos-weight-mode none
```

</details>

- **LoRA training** (`train_direct_lora.py` / `mlx_lm.lora`): see
  [docs/training.md](docs/training.md).
- **GEPA prompt evolution** (`run_gepa.py`, produces the prompt consumed by
  `m3_gepa`): see [docs/m3_m6.md](docs/m3_m6.md).

## Status

- ✅ **Measurement contour closed.** Group-aware split (question leakage 24.9% → 0),
  5×5 CV with the threshold fitted inside each fold, bootstrap CIs, and null
  calibration. A report without a CI fails schema validation.
- ✅ **One contract for every method.** All 19 are registered in
  `methods/registry.py` and reachable through `rag-judge`
  (`run`, `benchmark`, `eval`, `serve`, `list-methods`); 11 also run in the Gradio
  demo. Every run writes `scores.jsonl` + `report.json` + `run.yaml` with a git hash.
- ✅ **Method 6 measured and closed.** NLI grounding replaced SelfCheckGPT
  (~404k NLI pairs → ~52k, ~37 GPU-h → ~3 min). Best feature ROC-AUC
  0.548 [0.522; 0.575]; the NLI-model ablation changes nothing; stack delta negative.
  A reproducible negative result.
- ✅ **Method 3 diagnosed.** `m3.p_faith` is the only feature with a significant
  stacking gain (+0.0752, p = 0.013). Five improvement hypotheses — split axes,
  self-consistency, per-chunk verification, GEPA, fine-tuning — are implemented and
  wired end to end.
- ⏳ **Those five hypotheses have no numbers yet.** Earlier waves had no GPU, so the
  code was verified on dummy backends. `notebooks/standalone/method3_judge.ipynb`
  exists to close exactly that in one session.
- ⏳ Also open: baselines re-run over all 2233 cases so the stacking cohort (331)
  becomes comparable with the rest of the table, and a project-wide leaderboard.

## Documentation map

| Doc | What's inside |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Pipeline, module map, design decisions (conservative parsing, chat-template symmetry, 4-bit model, dependency pins) |
| [docs/data.md](docs/data.md) | Sample schema, marker vocabulary, dummy dataset, plugging in the real dataset |
| [docs/training.md](docs/training.md) | LoRA workflow, configs, why `--mask-prompt`, scaling up |
| [docs/lettucedetect.md](docs/lettucedetect.md) | LettuceDetect feature extraction + logistic regression |
| [docs/m3_m6.md](docs/m3_m6.md) | Selective Method 3/6 port from the `m3-m6` branch |
| [docs/experiments.md](docs/experiments.md) | All results so far, how to reproduce, environment gotchas |
| [docs/report/methods_3_6_results.md](docs/report/methods_3_6_results.md) | **Methods 3 and 6: what is measured, on which cohort, with which caveats** |
| [docs/report/article_materials_m3_m6.md](docs/report/article_materials_m3_m6.md) | Article source material for Methods 3 and 6: formulas, hyperparameters, every number with provenance, LaTeX-ready tables, and what is *not* measured |
| [docs/report/presentation_materials_m3_m6.md](docs/report/presentation_materials_m3_m6.md) | Slide-by-slide material for the team deck, including the claims on the current deck that no longer hold |
| [docs/report/wave3.md](docs/report/wave3.md) | Wave 3 outcome: stacking numbers, the M6 decision, what was left undone |
| [docs/specs/](docs/specs) | Per-phase measurement specifications (protocol, Method 3, Method 6, DataSphere runbook) |
| [docs/datasphere.md](docs/datasphere.md) | Running on Yandex DataSphere: hardware, stack pins, Jobs |
| [docs/handoff/](docs/handoff/HANDOFF.md) | Task cards and the working agreement between contributors |
| [docs/diagrams/](docs/diagrams/README.md) | PlantUML architecture, benchmark pipeline, method taxonomy, and sample data-flow diagrams |
| [notebooks/standalone/](notebooks/standalone/README.md) | Self-contained walkthroughs of Methods 3 and 6 |
| [notebooks/](notebooks/README.md) | DataSphere launcher notebooks (`00`–`40`) and full-FT Qwen2.5-7B notebooks |

## Project layout

```
data/alfa.jsonl         canonical corpus — 2233 cases, 72.4% reliable
data/splits/            group-aware folds; the only source of the split
data/dummy.jsonl        36 synthetic Russian banking RAG examples
from_organizators/      task statement, label scales, organizer baseline notebook
configs/                judge prompts (versioned), marker glossary, LoRA configs
src/rag_reliability/    schema, splits, evaluation protocol, stacking, metrics,
                        method packages (m3, m6, encoder, surface, ...),
                        method registry, rag-judge CLI
scripts/                CLI entry points — run from repo root
notebooks/              DataSphere launchers; standalone/ holds the method walkthroughs
jobs/                   DataSphere Job configs for runs longer than two hours
predictions/            scores.jsonl + report.json + run.yaml per method and variant
tests/                  unit tests (no GPU, no MLX required)
docs/                   specs, reports, task cards, diagrams
results/                checkpoints, adapters, intermediate artifacts (gitignored)
```
