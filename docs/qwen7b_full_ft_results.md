# Qwen2.5-7B full fine-tune — results

Full fine-tuning (not LoRA) of `Qwen/Qwen2.5-7B-Instruct` as a RAG-reliability
judge, for **Method 1 (`direct`)** and **Method 2 (`marker`)**. Trained on Yandex
DataSphere (1× A100 80 GB) with the notebooks under [`notebooks/`](../notebooks/README.md).

## Setup

- **Data**: the organizer dataset, `2245` samples, split `1796 / 224 / 225`
  (train / val / test), stratified by `reliable`, `seed=42`. Test metrics below
  are on the held-out `225`.
- **Training**: TRL `SFTTrainer`, bf16, gradient checkpointing, 8-bit AdamW,
  `assistant_only_loss=True` (loss on assistant tokens only — mirrors the MLX
  LoRA `--mask-prompt`). `EPOCHS=2`, `MAX_SEQ_LEN=2048`.
- **Eval**: greedy generation → repo `parsing.parse_prediction` →
  `metrics.evaluate_predictions` (same contract as `scripts/evaluate.py`).
  Reproduce with [`notebooks/eval_finetuned_datasphere.ipynb`](../notebooks/eval_finetuned_datasphere.ipynb).

## Models (Hugging Face Hub)

| name | Hub id | recipe |
|---|---|---|
| direct | `andy-takker/qwen2.5-7b-rag-judge-direct` | LR 1e-5, unbalanced |
| **direct_balanced** ⭐ | `andy-takker/qwen2.5-7b-rag-judge-direct-balanced` | LR 1e-5, **balanced train** |
| marker | `andy-takker/qwen2.5-7b-rag-judge-marker` | LR 1e-5, unbalanced |
| direct_balanced_lr3e5 | `andy-takker/qwen2.5-7b-rag-judge-direct-balanced-lr3e5` | LR 3e-5, balanced |

## Results (held-out test, 225 samples, macro-F1)

| metric | direct | **direct_balanced** | marker | direct_balanced_lr3e5 |
|---|---|---|---|---|
| reliable_f1_macro | 0.540 | **0.586** | 0.485 | 0.532 |
| faithfulness_f1_macro | 0.544 | **0.580** | 0.488 | 0.533 |
| relevance_f1_macro | 0.467 | **0.555** | 0.500 | 0.457 |
| invalid_output_rate | 0.000 | 0.000 | 0.000 | 0.000 |
| marker_f1_macro | — | — | 0.079 | — |

Baseline context (own splits, not directly comparable): tuned RuModernBERT
encoder ≈ `0.559` reliable macro-F1; base-Qwen `direct` prompt on the full
organizer set ≈ `0.495`.

## Findings

1. **Majority-class collapse from imbalance was the main problem.** The labels
   are skewed (`relevance` is ~88% positive; `reliable`/`faithfulness` ~72%).
   The unbalanced `direct` model predicted **every** test answer as relevant
   (relevance confusion `TN=0`), so relevance macro-F1 sat at `0.467`.
2. **Balancing the train split fixed it partially.** Oversampling rare
   `(faithfulness, relevance)` combos to the majority combo size (val/test
   untouched) lifted every metric — reliable `0.540 → 0.586`, relevance
   `0.467 → 0.555`, faithfulness `0.544 → 0.580`. `direct_balanced` beats the
   encoder baseline. Minority recall improved but stays modest (catches ~32% of
   unreliable, ~11% of irrelevant) — duplication-oversampling has limited power.
3. **LR 1e-5 was right; 3e-5 degraded.** Raising the learning rate pushed the
   model into noise, not signal — every metric dropped and the relevance
   collapse returned (`TN=0`). The lever was balancing, not LR.
4. **The marker error-typing does not work.** `marker_f1_macro ≈ 0.08` here and
   across every prior attempt in `results/` — the model almost always emits
   `none`/`unknown`. Its binary faithfulness/relevance are also weaker than
   `direct`. Use `direct`, drop the marker.
5. **Full FT of 7B is roughly on par with a small encoder** (`0.586` vs
   `0.559`) and far more expensive — the task appears to cap near
   `macro-F1 ≈ 0.56–0.59` for every method tried, i.e. the ceiling is the data,
   not the model size.

## Conclusion

**`direct_balanced` is the production judge** (reliable macro-F1 `0.586`), the
best result across all methods and baselines. Cheap hyperparameter levers are
exhausted (balance helped, higher LR hurt, marker fails). Further gains need
more or cleaner labeled data rather than a bigger model or more tuning.

**Reproduce the winner**: `notebooks/qwen7b_full_finetune_datasphere.ipynb` with
`MODE="direct"`, `BALANCE_TRAIN=True`, `LR=1e-5`, `EPOCHS=2`.
