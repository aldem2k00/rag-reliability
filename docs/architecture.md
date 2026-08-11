# Architecture

The prompt-based pipeline is a straight line; every dummy, zero-shot, and LoRA
judge experiment walks the same five steps, so results are comparable:

```
data (jsonl) ──► prompt formatting ──► inference ──► parsing ──► evaluation
  dataset.py       prompts.py         scripts/      parsing.py   metrics.py
                   formatting.py
```

The LettuceDetect method shares the same `RagSample` input and `Prediction`
output contracts, but replaces prompt generation/parsing with feature
extraction and a classifier:

```
data (jsonl) ──► feature extraction ──► classifier ──► evaluation
  dataset.py       methods/lettucedetect  Prediction    metrics.py
```

`scripts/run_benchmark.py` is the shared operational interface. It runs each
supported method into the same artifact layout:

```
results/<benchmark>/<method>/predictions.jsonl ──► metrics.json
```

`scripts/serve_demo.py` is the manual local UI. It accepts one
question/context/answer triple, runs a selected method, and displays the
standard `Prediction` payload plus optional gold-label correctness.

## Modules (`src/rag_reliability/`)

| Module | Responsibility |
|---|---|
| `schema.py` | Pydantic models: `RagSample` (input + gold labels), `Prediction` (parsed model output), `EvaluationResult` (metrics). `reliable = faithfulness AND relevance` is a derived property on both sides. |
| `prompts.py` | English judge prompts for both modes; prompts handle either a single question or a full dialog in the `QUESTION` section. |
| `formatting.py` | Builds SFT targets and `{"prompt", "completion"}` training records; `resolve_marker()` implements the `none`/`unknown` fallback used by both training and evaluation. |
| `parsing.py` | Raw LLM text → `Prediction`. Three-stage fallback: balanced-JSON extraction → regex → conservative `(0, 0, invalid_output=True)`. Never raises on model output. |
| `metrics.py` | Macro-F1 for reliable/faithfulness/relevance; marker F1 + confusion (marker mode only). Joins predictions to samples by `id`, raises on missing ids. |
| `dataset.py` | JSONL IO, stratified 80/10/10 split by `reliable` (seed=42), training-file writer. |
| `dummy_model.py` | Deterministic pseudo-LLMs so the whole pipeline runs without a model (see [experiments.md](experiments.md)). |
| `methods/lettucedetect/features.py` | LettuceDetect detector setup, token-score aggregation, feature extraction. |
| `methods/lettucedetect/classifier.py` | sklearn classifier helpers and conversion to `Prediction`. |

## Scripts (`scripts/`)

| Script | Role |
|---|---|
| `run_prompt_baseline.py` | Zero-shot judge over a dataset; `--backend dummy` or `--backend mlx`. |
| `infer.py` | Same as the mlx baseline but loads a trained LoRA adapter (`--adapter-path`). Output format is identical, so `evaluate.py` works for both. |
| `infer_lettucedetect.py` | Runs a trained LettuceDetect logistic-regression classifier and writes standard predictions. |
| `evaluate.py` | Predictions + gold → metrics json. |
| `run_benchmark.py` | Unified runner for dummy, prompt, LoRA, LettuceDetect, encoder, Method 3, and Method 6 methods; every method is normalized to `Prediction` JSONL before evaluation. |
| `serve_demo.py` | Local Gradio UI for manually running one sample through a selected method. |
| `train_direct_lora.py` / `train_marker_lora.py` | Prepare SFT splits and print the exact `mlx_lm.lora` command (they do not train themselves). |
| `train_lettucedetect.py` | Extracts LettuceDetect features and trains the logistic-regression classifier. |
| `prepare_data.py` | Raw dataset → `RagSample` jsonl (see [data.md](data.md)). |

## Methods

`schema.py::ALLOWED_MARKERS` is the single source of truth for the marker
vocabulary. It includes both the original compact markers used by the dummy
dataset and the official organizer `reason_*` markers.

- **Method 1 — direct** (`mode=direct`): model outputs
  `{"faithfulness": 0|1, "relevance": 0|1}`.
- **Method 2 — marker** (`mode=marker`): model first names the error type,
  then the labels: `{"marker": "...", "faithfulness": 0|1, "relevance": 0|1}`.
  Hypothesis: forcing an error-type decision improves label quality and gives
  diagnosable failure categories for free.
- **LettuceDetect baseline**: LettuceDetect produces token-level
  unsupported probabilities over the answer; `max`, `mean`, and fraction above
  threshold are fed into a multi-output logistic regression for faithfulness
  and relevance.
- **Method 3 — LLM judge** (`m3_*`): a prompt judge with zero-shot, few-shot,
  and GEPA-prompt modes. The OpenAI-compatible logprob variant turns PASS/FAIL
  verdict token probabilities into faithfulness/relevance probabilities for
  validation-fitted threshold evaluation.
- **Method 6 — SelfCheck** (`m6_selfcheck`): generates answer samples, derives
  NLI contradiction and semantic-entropy features (plus relevance), then maps
  those features to labels with explicit thresholds in the Method 6 pipeline.

## Design decisions

- **Conservative parsing.** An unparseable output counts as
  `faithfulness=0, relevance=0` and increments `invalid_output_rate`. A judge
  that produces garbage must not look reliable. This applies to prompt-based
  methods; LettuceDetect and Method 6 write structured `Prediction` objects
  directly.
- **Chat template symmetry.** Both training (`mlx_lm` `CompletionsDataset`
  applies the tokenizer chat template to prompt/completion pairs) and
  inference (`apply_chat_template` in the scripts) wrap prompts identically —
  verified, not assumed.
- **`--mask-prompt` in training.** The judge prompt is ~50× longer than the
  JSON completion; without prompt masking the loss is dominated by prompt
  tokens.
- **4-bit base model.** `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (~840 MB)
  instead of bf16 (~3 GB): faster download, ~8 s inference over 36 samples,
  QLoRA-style training works out of the box.
- **`transformers<5` pin.** transformers 5.x breaks `mlx-lm` at import time
  (`TOKENIZER_MAPPING.register` receives a `str`); pinned in the `mlx` extra.
