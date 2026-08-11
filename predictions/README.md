# Predictions ported from the m3-m6 branch

Prediction artifacts produced on the original `m3-m6` branch and copied here
verbatim for reference. They use the original platform contract — JSONL rows
`{"id": str, "p_faith": float, "p_rel": float, "meta": {...}}` (probabilities,
not the binary `Prediction` schema of this repository) — with a `run.yaml`
(config + git hash + seed) and `report_*.json` (metrics with thresholds picked
on val) next to each run.

Layout:

- `pseudo_debug/` — debug runs on the synthetic SberQuAD-based pseudo-corpus
  through a cloud OpenAI-compatible provider (Method 3 zero/few-shot and GEPA
  prompts, Method 6 baseline). These runs contain only 20–30 synthetic cases.
  Their metrics are not comparable with Alfa results or with one another as
  production-quality estimates. In particular, the recorded `0.6607`
  zero-shot “test” score is withdrawn from result comparisons: it was measured
  on 30 synthetic examples and will be replaced by canonical 2233-case CV
  reports described in `docs/specs/10_PHASE0_измерительный_контур.md`.
- `alfa/m3/` — stage-1 runs of Method 3 on the Alfa corpus through
  OpenRouter (explicit data-owner opt-in), including the 72B backbone ablation.
- `alfa/baselines/` — surface/majority/encoder baselines computed locally.

GEPA prompt-evolution reports for these runs live in `docs/gepa_reports/`.
