# Predictions ported from the m3-m6 branch

Prediction artifacts produced on the original `m3-m6` branch and copied here
verbatim for reference. They use the original platform contract — JSONL rows
`{"id": str, "p_faith": float, "p_rel": float, "meta": {...}}` (probabilities,
not the binary `Prediction` schema of this repository) — with a `run.yaml`
(config + git hash + seed) and `report_*.json` (metrics with thresholds picked
on val) next to each run.

Layout:

- `cloud/` — debug runs on the synthetic pseudo-corpus through a cloud
  OpenAI-compatible provider (Method 3 zero/few-shot and GEPA prompts, Method 6
  baseline). Debug-only numbers; never mixed with local results.
- `alfa_openrouter/` — stage-1 runs of Method 3 on the Alfa corpus through
  OpenRouter (explicit data-owner opt-in), including the 72B backbone ablation.
- `local/baselines/` — surface/majority/encoder baselines computed locally.

GEPA prompt-evolution reports for these runs live in `docs/gepa_reports/`.
