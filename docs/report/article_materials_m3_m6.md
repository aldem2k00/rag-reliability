# Article materials: Method 3 (LLM judge) and Method 6 (NLI grounding)

Source material for the SMILES-2026 article, shaped to match
`reliability_lettucedetect_nli_article.tex`. Everything here is extracted from the
code and artifacts in this repository; each number carries its provenance so a claim
can be checked without rerunning anything.

Written in English because the article is in English — sentences are meant to be
liftable directly. Prompts are quoted in Russian, as they are in the code.

> **Status warning, read before drafting.** Method 6 is measured end to end.
> Method 3 has a full diagnosis and a significant stacking contribution, but its five
> improvement hypotheses are **implemented and not yet run** — no GPU was available in
> the waves that built them. Section 8 lists exactly which cells of which table are
> still empty. Do not let a draft imply otherwise.

---

## 1. One-paragraph summary of each method

**Method 3 — LLM-as-judge with prompt optimization.** A local
`Qwen/Qwen2.5-7B-Instruct` served by vLLM reads a case and emits a `PASS`/`FAIL`
verdict per axis. The score is not the sampled label: it is `P(PASS)` recovered from
the token logprobs at the verdict position, which makes the method a continuous
ranker whose threshold is owned by the evaluation protocol. Five variants are
implemented: holistic single call, split axes, self-consistency over k samples,
per-chunk verification, and a GEPA-evolved prompt; plus a LoRA fine-tuned counterpart.

**Method 6 — NLI grounding.** No LLM at inference. The answer is split into sentences
with `razdel`, every (chunk, sentence) pair is scored by a multilingual NLI
cross-encoder, and the resulting matrix is reduced to 8 grounding features plus 4
coverage features. The method never binarizes: the protocol fits the threshold.

Both feed the same contract: `scores.jsonl` (continuous features) →
`scripts/evaluate_cv.py` → `report.json` with a 95% CI.

---

## 2. Data (differs from the template's numbers — check before reuse)

The template uses the organizer's 1,796/224/225 stratified split over 2,245 rows.
This repository uses a **deduplicated corpus of 2,233 cases and a group-aware split**,
because the stratified split leaked.

| Property | Value | Source |
|---|---|---|
| Cases | 2,233 | `data/alfa.jsonl` |
| faithfulness = 1 | 1,642 (73.53%) | computed |
| relevance = 1 | 1,961 (87.82%) | computed |
| reliable = 1 | 1,616 (72.37%) | computed |
| Chunks per case | median 5, mean 5.75, range 5–8 | computed |
| Answer length | median 418 chars, mean 427 | computed |
| Context length | median 6,410 chars, mean 6,655 | computed |
| Dialogue length | median 458 chars | computed |
| Sentences per answer | median 4, mean 4.05, max 12; 9,053 total | `m6_grounding/base/scores.jsonl` |

### 2.1 Label contingency — an important contrast with the template

| | relevance = 0 | relevance = 1 |
|---|---:|---:|
| **faithfulness = 0** | 246 | 345 |
| **faithfulness = 1** | 26 | 1,616 |

The template's split has "only one row faithful but irrelevant", which makes relevance
almost powerless. On this corpus the asymmetry is milder but points the same way:
**26 rows are faithful-but-irrelevant** (the only rows a relevance component can fix),
against **345 unfaithful-but-relevant** rows. So relevance can correct at most 26 of
617 unreliable cases, while every relevance false negative on the 1,616 reliable rows
damages the conjunction directly. This is the single most useful framing for the
Discussion section, and it is quantitative here rather than anecdotal.

### 2.2 Marker distribution (13 organizer codes)

| Marker | n | share |
|---|---:|---:|
| `none` (reliable) | 1,616 | 72.37% |
| `unknown` | 477 | 21.36% |
| `reason_hallucinated_fact` | 32 | 1.43% |
| `reason_answer_for_operator` | 23 | 1.03% |
| `reason_off_topic_answer` | 18 | 0.81% |
| `reason_chunk_fact_mixup` | 18 | 0.81% |
| `reason_false_verification` | 14 | 0.63% |
| `reason_irrelevant_chunk_used` | 14 | 0.63% |
| `reason_incomplete_answer` | 6 | 0.27% |
| `reason_wrong_navigation` | 5 | 0.22% |
| `reason_outdated_fact` | 4 | 0.18% |
| `reason_missed_chunk_conditions` | 3 | 0.13% |
| `reason_reveals_ai_identity` | 2 | 0.09% |
| `reason_missed_complaint_handoff` | 1 | 0.04% |

Worth stating in the paper: **21.36% of cases are unreliable with an `unknown` marker**,
so per-marker analysis is only possible on ~6% of the corpus. Any claim of the form
"the method detects error type X" is underpowered by construction.

### 2.3 Split: why group-aware, and what it costs

`data/splits/folds_alfa.json`, built by `src/rag_reliability/splits.py`.

- Grouping is a union-find over three edge sources: identical normalized last client
  utterance; cosine ≥ 0.9 over char 3–5-gram TF-IDF of the dialogue; identical first
  retrieved chunk. Group keys are content-addressed (sha1 of member ids), so the
  assignment is stable across runs.
- Result: 715 groups, leakage `query_overlap = 0.0`, `near_dup_0.9 = 0.0`,
  `chunk1_overlap = 0.0`.
- **One group holds 753 ids** (33.7% of the corpus — the same question asked over and
  over). It cannot be placed in any fold without dominating it, so it is excluded.
  Every CV number in this repository is therefore computed on **n = 1,480**, while
  ROC-AUC over raw features (which needs no split) uses all 2,233.
- Under the previous stratified split the leakage was 24.9% by question, and a 1-NN
  memorizer scored 0.6278 — higher than any real method. That single fact is the
  strongest available justification for the protocol change and belongs in the paper.

### 2.4 Protocol

5 folds × 5 repeats. For each fold the decision threshold is fitted **only on the
training part** by scanning a 0.01 grid over [0,1] and taking the smallest threshold
that maximizes macro-F1 (ascending scan, replacement on strict improvement only).
Out-of-fold decisions are glued and scored once. Uncertainty: percentile bootstrap
over cases, B = 10,000. Noise floor: 500 null runs (label permutation through the same
fit-and-apply procedure); `above_noise` is true when the observed value exceeds the
95th percentile of the null distribution. `EvaluationReport` refuses to serialize a
primary metric without `ci95`.

**Baseline floor.** Constant "always reliable" on the 1,480-case cohort gives
macro-F1 = **0.4203** (computed analytically; a constant predictor has no threshold,
so CV does not apply and a CI is undefined).

---

## 3. Method 6 — full method description

### 3.1 Why not SelfCheckGPT (motivation worth a paragraph)

The branch was scoped as SelfCheckGPT: sample N alternative answers from the same
model and measure disagreement with the original. It was stopped on a validity
argument, not a performance one: **the production bot's prompt is unavailable**, so
any generator placed in the loop is a *different system*, and the measured disagreement
would quantify the gap between two models rather than the bot's hallucination. The
metric would be uninterpretable regardless of its value.

The replacement is direct NLI grounding — premise = a retrieved chunk, hypothesis = a
sentence of the answer — which removes the generator entirely.

| | SelfCheckGPT | NLI grounding |
|---|---:|---:|
| NLI pairs per corpus | ~404,000 | **51,763** (measured) |
| NLI forward windows | — | 69,478 (measured) |
| LLM generations | 5–20 per case | 0 |
| Wall clock | ~37 GPU-h (estimate) | **~3 min** on A100 |

Pair and window counts are recorded in
`predictions/alfa/m6_grounding/base/run.yaml` (`nli_pairs`, `nli_windows`).

### 3.2 Pipeline

1. **Sentence segmentation** — `razdel`, not a period regex. The regex split
   "макс.", "т.д.", "0.5%" and numbered steps into fragments; a fragment like "1." is
   entailed by nothing and depressed `min_entail` artificially.
2. **NLI matrix** — for sentences $h_1..h_m$ and chunks $c_1..c_n$, a
   sequence-classification cross-encoder produces entailment/neutral/contradiction
   logits per pair. Two details matter:
   - **Two-class softmax** over {entail, contra}, neutral discarded. On long banking
     premises the three-class normalization gives neutral most of the mass and
     contradiction stops being a signal at all. (SelfCheckGPT-NLI convention.)
   - **Premise windowing.** The hypothesis budget is measured first; the premise is
     never truncated, it is split into overlapping windows (`max_length` 512, overlap
     128, 4 tokens reserved for special tokens). Windows are collapsed by taking the
     window with maximal entailment, and `contra` is read **from that same window** —
     independent per-class maxima describe a window that does not exist and break the
     normalization.
3. **Aggregation.** Per sentence, over chunks: $q_j=\max_i e_{ij}$ (one supporting
   source is enough) and $\bar c_j=\max_i c_{ij}$ (contradicting any source counts).

### 3.3 The twelve features

Grounding (`src/rag_reliability/methods/m6/grounding.py`), with observed
distributions over 2,233 cases:

| Feature | Definition | median | mean | sd |
|---|---|---:|---:|---:|
| `m6.max_entail` | $\max_j q_j$ | 0.997 | 0.984 | 0.042 |
| `m6.min_entail` | $\min_j q_j$ — the weakest link | 0.751 | 0.678 | 0.274 |
| `m6.mean_entail` | $\frac1m\sum_j q_j$ | 0.902 | 0.868 | 0.120 |
| `m6.mean_contra` | $\frac1m\sum_j \bar c_j$ | — | — | — |
| `m6.max_contra` | $\max_j \bar c_j$ | — | — | — |
| `m6.frac_unsupported` | $\frac1m\sum_j \mathbb{I}(q_j<0.5)$ | 0.000 | 0.077 | 0.149 |
| `m6.n_sentences` | $m$ | 4.0 | 4.05 | — |
| `m6.chunk_spread` | $|\{\arg\max_i e_{ij}\}|$ — distinct source chunks | 3.0 | 2.663 | 0.872 |

`min_entail` is the target feature: one unsupported sentence makes the whole answer
unfaithful, and the mean washes that out. `chunk_spread` was designed as a signature
of `reason_chunk_fact_mixup` (a fact welded from two different chunks).

Coverage (`coverage.py`) — 4 features, **reusing the same matrix, zero extra NLI
calls**. Conditions are searched only inside chunks that grounding marked as sources
($\arg\max$ of the entailment row); scanning all 5–8 chunks turns the metric into
noise because most conditions are unrelated to the question. An empty source set
raises an error rather than reporting full coverage.

| Feature | Definition | median | mean |
|---|---|---:|---:|
| `m6.n_conditions` | conditions found in source chunks | 7.0 | 8.73 |
| `m6.cond_coverage` | share reflected in the answer | 0.600 | 0.543 |
| `m6.digit_coverage` | share of source numbers present in the answer | — | — |
| `m6.uncovered_max` | max weight among uncovered conditions | — | — |

Condition markers (regex, case-insensitive): `если`, `при условии`, `не менее`,
`не более`, `до \d+`, `от \d+`, `в течение`, `не позднее`, `только для`, `кроме`,
`минимальн`, `максимальн`, `\d+([.,]\d+)?\s*%`, `\d+\s*(руб|₽|дн|мес|год|час)`.
A 30-character lookahead attaches the numeric core to its marker ("не менее 100 руб."),
markers closer than 8 characters merge into one condition, and a condition with a
numeric core weighs 1.0 against 0.5 for a purely lexical one.

### 3.4 Models

- Default: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, fp16 on CUDA,
  batch 64, `max_length` 512, overlap 128.
- Ablation: `Feudor2/rumodernbert-nli` (Russian-trained), same settings.

### 3.5 Results — ROC-AUC of every feature, n = 2,233, B = 10,000

Full corpus, no split needed (feature ranking, not a fitted decision). Axis =
faithfulness; the `reliable` column is the point estimate only.

| Feature | vs faithfulness | vs reliable |
|---|---|---:|
| `m6.mean_entail` | 0.556 [0.529; 0.583] | 0.554 |
| **`m6.min_entail`** (target) | **0.548 [0.522; 0.575]** | 0.547 |
| `m6.max_entail` | 0.525 [0.498; 0.552] | 0.520 |
| `m6.chunk_spread` | 0.510 [0.485; 0.535] | 0.507 |
| `m6.n_sentences` | 0.507 [0.481; 0.533] | 0.502 |
| `m6.cond_coverage` | 0.506 [0.479; 0.532] | 0.503 |
| `m6.max_contra` | 0.499 [0.473; 0.526] | 0.496 |
| `m6.n_conditions` | 0.494 [0.467; 0.521] | 0.491 |
| `m6.mean_contra` | 0.491 [0.464; 0.518] | 0.493 |
| `m6.digit_coverage` | 0.483 [0.456; 0.510] | 0.482 |
| `m6.uncovered_max` | 0.473 [0.450; 0.495] | 0.475 |
| `m6.frac_unsupported` | 0.461 [0.440; 0.483] | 0.465 |

Source: `predictions/alfa/m6_grounding/base/auc.json`.

Reading: twelve intervals, none further than 0.06 from 0.5. Only two features have a
lower bound above 0.5 (`mean_entail`, `min_entail`). Values below 0.5 are inverted
signal of the same magnitude — expected for `frac_unsupported`, which grows with
unfaithfulness.

### 3.6 NLI-model ablation

| Feature | mDeBERTa-v3-xnli | ruModernBERT-nli |
|---|---|---|
| `m6.mean_entail` | 0.556 [0.529; 0.583] | 0.557 [0.529; 0.584] |
| `m6.min_entail` | 0.548 [0.522; 0.575] | 0.556 [0.529; 0.583] |
| `m6.max_entail` | 0.525 [0.498; 0.552] | 0.533 [0.507; 0.559] |
| `m6.max_contra` | 0.499 [0.473; 0.526] | 0.451 [0.424; 0.478] |
| `m6.mean_contra` | 0.491 [0.464; 0.518] | 0.453 [0.426; 0.480] |
| `m6.frac_unsupported` | 0.461 [0.440; 0.483] | 0.452 [0.429; 0.475] |

Intervals coincide almost point for point on the entailment features: **the choice of
NLI checkpoint does not decide anything here.** Interesting secondary observation: the
Russian model is *worse* on the contradiction features (0.451 vs 0.499), consistent
with contradiction being the weaker of the two heads on this data.

### 3.7 Row-level result and stack contribution

| Configuration | macro-F1 | 95% CI | above noise | n |
|---|---:|---|---|---:|
| `m6.min_entail`, mDeBERTa | 0.5339 | [0.5070; 0.5601] | yes (percentile 99.8) | 1,480 |
| `m6.min_entail`, ruModernBERT | 0.5397 | [0.5128; 0.5658] | yes (percentile 100.0) | 1,480 |
| `independent` (rules) | 0.5122 | [0.4866; 0.5382] | no | 1,480 |
| `surface` (text baseline) | 0.5350 | [0.5102; 0.5615] | yes | 1,480 |

Grounding is indistinguishable from the surface baseline. In the stack every M6
feature makes things worse:

| Stack configuration | macro-F1 | 95% CI | Δ vs surface | Δ 95% CI | p |
|---|---:|---|---:|---|---:|
| surface (base, 331-case cohort) | 0.5791 | [0.5222; 0.6348] | — | — | — |
| + `m6.min_entail` | 0.5686 | [0.5128; 0.6256] | −0.0105 | [−0.0547; +0.0348] | 0.657 |
| + `m6.mean_entail` | 0.5678 | [0.5122; 0.6230] | −0.0114 | [−0.0528; +0.0317] | 0.617 |
| + `m6.frac_unsupported` | 0.5654 | [0.5097; 0.6209] | −0.0137 | [−0.0537; +0.0273] | 0.524 |

Source: `docs/report/wave3.md` §3.1. Only three of the twelve features were entered
into the ablation on purpose — running all twelve is selection optimism; none of the
three came close enough to zero for the remaining nine to change the conclusion.

### 3.8 Decision rule, pre-registered

Fixed **before** the run (task card C4): AUC ≥ 0.60 → develop as a standalone method;
0.53–0.60 → hand the features to the stack, do not develop the branch; ≤ 0.53 → close.
Observed: 0.548, lower bound 0.522 → middle band. The band prescribes testing an
alternative NLI model; that was done and changed nothing.

**Outcome: the branch is closed as a development direction.** This is a measured
negative result, reproducible in 15 minutes — the paper should present it as such,
not as an absence of results.

### 3.8b The prior SelfCheckGPT results, and why none of them survive

The branch published three claims before the redefinition. All three were audited in
`docs/specs/40_PHASE3_метод6.md` §0 and none can be carried into the paper as stated.
This audit is itself reportable material — it is a clean example of a measurement
artifact being mistaken for a finding.

| Prior claim | What it actually was |
|---|---|
| "M6 reaches reliable macro-F1 = 0.596" | On a 20-case pseudo-corpus, the faithfulness branch degenerates to a constant predictor: threshold `t_faith = 0.01` against a minimum score of 0.4495 marks all 20 cases reliable. `f1_macro_faith = 0.3939… = 13/33` is exactly the constant-predictor value. The published 0.596 is produced by a supervised relevance logistic regression trained on those same 20 cases — **the SelfCheck signal enters no published number.** |
| "Semantic entropy inverts at N = 10 (−0.164) — alignment collapse" | The delta is **0.38 standard errors** (SE ≈ 0.433 at n = 20). Its sign flips with the clustering threshold in the team's own ablation: at N = 5 it is −0.065 (thr 0.5), +0.017 (thr 0.4), +0.065 (thr 0.3). The plug-in entropy estimator's bias at N = 6 vs N = 11 reproduces the "inversion" mechanically. Correct wording: *no semantic-entropy signal was detected at the available power.* |
| "Contradiction is the workhorse: Δ = +0.939" | Measured on the **synthetic** corpus, not on real data. |

A fourth defect is worth one line because it explains the shape of the old pipeline:
the hard-coded `contradiction_threshold = 0.5` sat against a feature median of 0.0099,
so the condition was true almost always and faithfulness was in practice decided by
entropy alone.

Two further arguments against the original formulation, beyond the proxy-generator
problem in §3.1: SelfCheckGPT is designed as a *zero-resource* method for settings
**without** context, whereas here the context exists and the sampler even sees it, so
consistency is largely subsumed by direct entailment against the context; and the
literature reports that consistency features reach AUROC ≈ 0.556 and **change sign**
across model families in grounded settings.

### 3.9 Hypothesis H4

"SelfCheckGPT descendants give useful signal on faithfulness but lose to supervised
methods on relevance."

Answer from the data: **partially, and weaker than the hypothesis states.** Signal on
faithfulness is distinguishable from a coin flip, but only for 2 of 12 features, and
its magnitude (AUC 0.548–0.556) is such that the difference *between the axes* lies
inside the confidence intervals. Mean AUC across the twelve features is 0.504 against
faithfulness and 0.499 against relevance — a gap of 0.005 with interval widths of
±0.027. The honest formulation: the data neither confirm nor refute the axis
asymmetry.

---

## 4. Method 3 — full method description

### 4.1 Verdict probability from logprobs (the mechanism worth explaining)

The judge is asked for three lines per axis: `ANALYSIS`, `MARKER`, and
`FAITHFULNESS/RELEVANCE: PASS|FAIL`. The score is not the parsed label:

1. Locate the **last** `AXIS:` anchor in the token stream (regex over the concatenated
   token text), then find the first whole or BPE-merged `PASS`/`FAIL` token inside that
   axis section. Anchoring matters: an earlier version took the first two PASS/FAIL
   occurrences in the stream, which caught mentions inside the free-form analysis.
2. At the verdict position, read the top-logprobs. If both sides are visible,
   $P(\text{PASS}) = \sigma_{\text{2-way}}$ = softmax over the two logprobs. If only
   one side is visible, its logprob is an absolute log-probability and is
   **exponentiated** — $P = e^{\ell_{\text{PASS}}}$ or $1 - e^{\ell_{\text{FAIL}}}$.
   Exact token matches beat prefixes; one-character prefixes are accepted only for
   `P` and `F` because local tokenizers split `PASS`/`FAIL` after the first letter.
3. If neither side is visible, the fallback is 0.5.

A historical bug is worth one sentence in the paper as a cautionary note: an earlier
implementation applied a **sigmoid to a logprob** instead of exponentiating it, which
inverts monotonicity — and a green unit test froze the behavior for three branches.
The verdict extractor now requires `--max-logprobs 25` on the vLLM side (the client
requests `top_logprobs=20`); without it the server returns verdicts without
probabilities and the whole method silently degenerates to regex parsing with all
scores equal to 0.5. The notebooks assert `prob_method == "logprobs"` on a 5-case
smoke run for exactly this reason.

### 4.2 Prompts (versioned, in `configs/prompts/`)

Both prompts quote the organizer's rubric **verbatim** rather than paraphrasing it.

**`faithfulness/v2`** (`needs_context: true`): the auditor role, the verbatim
definition of the label, the 4-step verification order from the annotation guide, and
11 explicit FAIL criteria (6 from the organizer rubric, 5 added to cover the marker
taxonomy — previously the prompt covered ~6 of the 13 codes). A checklist of all 13
marker codes from `configs/markers.yaml` is injected via a `{marker_checklist}`
placeholder. Output format: exactly three lines.

**`relevance/v2`** (`needs_context: false` — **the prompt receives no chunks at all**):
relevance is a relation between the dialogue and the answer, and context only adds
noise. It carries 6 FAIL criteria, the annotators' control question ("does the answer
substantively help solve the client's problem?"), and three verbatim examples showing
that an *incomplete but on-topic* answer is still PASS. Only 5 marker codes are
attached to this axis; `reason_incomplete_answer` is deliberately excluded because
incompleteness belongs to faithfulness. Cost: ~600 prompt tokens instead of ~3,000.

### 4.3 Variants

| Variant | What changes | Score keys | Budget (A100, 2,233 cases) |
|---|---|---|---|
| holistic (`joint`) | one call, both verdicts | `m3.p_faith`, `m3.p_rel` | ~15 min (6.7M prompt / 0.7M completion tokens) |
| split axes (`axes`) | two independent calls, relevance without chunks | same + `_vote`, `_std` | ~20 min (relevance axis alone ~5 min, 1.3M/0.4M) |
| self-consistency | k samples per axis at T>0 | `+ m3.p_*_vote`, `m3.p_*_std` | ~50 min at k=8 (6.7M prefix-cached / 5.4M) |
| per-chunk | one call per chunk, faithfulness only | 5 keys, see below | ~40 min (~7M / 2.2M) |
| GEPA | evolved instruction, same inference path | as `axes` | ~1.2 h per run, ~7 h for H5 |
| `ft_judge` | LoRA-tuned Qwen2.5-7B | `m3.p_faith`, `m3.p_rel` | ~1.5 h per fold, ~7.5 h for 5 folds |

**Self-consistency aggregation.** Probabilities are averaged, not votes:
$p=\frac1N\sum_t p_t$. Votes are kept alongside as `p_vote` (share of recovered PASS
verdicts; 0.5 if none recovered) and the spread `p_std` (population standard
deviation) is exposed as a feature in its own right — a judge that disagrees with
itself is a difficulty signal. Rationale for the paper: voting quantizes the score to
1/N and discards exactly the information the logprobs were extracted for. Raw model
responses are cached (not the extracted probabilities), so an ablation over
N ∈ {1,4,8,16} reuses prefixes of one N=16 sample set without new calls.

**Per-chunk features** (`m3/perchunk.py`), faithfulness axis only, support threshold
0.5, with $s_{(1)} \ge s_{(2)} \ge \dots$ the sorted per-chunk scores:

| Key | Definition |
|---|---|
| `m3.max_chunk_score` | $s_{(1)}$ — best supporting chunk |
| `m3.mean_chunk_score` | mean over chunks |
| `m3.chunk_disagreement` | $s_{(1)}-s_{(2)}$ — target feature, a direct detector of `chunk_fact_mixup` |
| `m3.n_supporting` | $|\{i: s_i > 0.5\}|$ |
| `m3.argmax_chunk` | 1-based index of the best chunk |

Note for the evaluation section: a per-chunk artifact has **no axis pair**, so the
axis diagnostics are skipped rather than fabricated from a foreign signal.

**GEPA.** DSPy-based instruction evolution seeded from the *production* axis prompt.
Metric = **balanced accuracy** of the axis, not accuracy: at a 72% positive base rate
a candidate improves accuracy simply by answering PASS more often, while the reported
macro-F1 degrades; balanced accuracy gives a constant-PASS candidate exactly 0.5 at
any imbalance. $D_{\text{pareto}} = 300$ cases stratified by the axis label, drawn
only from the training part of the current fold — held-out never enters optimization.
Budget `auto=medium`, class weights computed on $D_{\text{pareto}}$ and normalized so
per-example scores stay in [0,1].

**Fine-tuned judge.** LoRA, r = 256, α = 512, all-linear target modules, lr 2e-4,
3 epochs, `max_length` 2,048, batch 1 × grad-accum 8, seed 42, `save_strategy=epoch`.
Class imbalance is handled twice — `pos_weight_mode=balanced` plus negative
oversampling — because an earlier 1.5B run collapsed to a constant (1,1) verdict under
the 72/28 imbalance. Training is per fold; the five held-out parts are concatenated
into one OOF artifact, so every case is predicted by a model that never saw it.

### 4.4 Diagnosis of the baseline judge (measured, and the origin of every hypothesis)

| Indicator | Value | Interpretation |
|---|---:|---|
| AUC(`p_faith`) vs faithfulness | 0.627 | weak but real signal |
| **AUC(`p_rel`) vs relevance** | **0.497** | no signal at all |
| Share of `RELEVANCE: PASS` verdicts | 98.6–100% | gold share is 84.9%: the judge nearly always says yes |
| Share of `p_faith > 0.99` | 50–79% | saturation — nothing left to rank |
| Recall of the unreliable class | 0.22–0.33 | misses two thirds of bad answers |
| `p_faith` with 8 chunks vs 5 | 0.857 vs 0.826 | wrong direction: more context, more confidence |

Legacy holdout numbers (single split, n = 223, **not comparable** with the CV table):
zero-shot 0.5841, few-shot 0.5309, GEPA-markers 0.5502, surface 0.5982.

### 4.4b Measured saturation and extraction, per artifact

Computed directly from the committed `scores.jsonl` files, so these are reportable
without a rerun. `invalid_output` is 0 on every run — the parser never failed.

| Run | n | invalid | median `p_faith` | share `p_faith` > 0.99 | share `p_rel` > 0.5 |
|---|---:|---:|---:|---:|---:|
| `m3/zero_shot` (7B) | 446 | 0 | 0.995 | **0.540** | 0.989 |
| `m3/few_shot` (7B) | 446 | 0 | 1.000 | **0.758** | 0.998 |
| `m3/gepa_markers_s0` (7B) | 223 | 0 | 0.996 | 0.583 | 0.960 |
| `m3/zero_shot_72b` | 223 | 0 | 0.900 | 0.000 | 0.955 |
| `m3/few_shot_72b` | 223 | 0 | 0.900 | 0.000 | 0.951 |

Two things to take from this table.

**Saturation is real and few-shot makes it worse.** Zero-shot puts 54.0% of cases above
0.99; few-shot pushes that to 75.8%. Adding demonstrations made the judge more
confident, not more discriminative — which is the direct motivation for the
self-consistency hypothesis, and a result in its own right.

**The 72B artifacts are degenerate and must not be reported as judge scores.**
`zero_shot_72b` has exactly **two distinct values** of `p_faith` across 223 cases:
0.9 (190 cases) and 0.1 (33). That is the regex-fallback convention, not a probability
distribution — the endpoint returned no usable logprobs. Any 72B number in older
tables (e.g. "zero_shot_72b val 0.5894") describes a binary verdict mapped to two
constants, and the ranking metrics computed on it are meaningless. This is the exact
failure mode the mandatory logprob smoke test was introduced to prevent.

For contrast, `zero_shot` (7B, local vLLM) has 82 distinct rounded values of `p_faith`.

### 4.4c Few-shot configuration

`configs/few_shot.yaml` holds **7 examples drawn from the real training split** (never
val/test), with hand-written `analysis` fields and case ids in comments. Coverage is
deliberate: 2 reliable, `hallucinated_fact`, `incomplete_answer`,
`answer_for_operator`, `chunk_fact_mixup`, and one (faith = 1, rel = 0) anchor case —
the cell that teaches the judge the axes are independent. Contexts are trimmed to the
relevant fragments; PII was already masked by the curators (`[NAME]`, `[NUMBER]`,
`[URL]`).

### 4.5 The one significant result in the project

| Stack configuration | macro-F1 | 95% CI | Δ vs surface | Δ 95% CI | p |
|---|---:|---|---:|---|---:|
| surface (base) | 0.5791 | [0.5222; 0.6348] | — | — | — |
| **+ `m3.p_faith`** | **0.6543** | [0.5956; 0.7097] | **+0.0752** | [+0.0158; +0.1344] | **0.013** |
| + `m3.p_rel` (diagnostic) | 0.5676 | [0.5108; 0.6228] | −0.0115 | [−0.0344; +0.0102] | 0.338 |
| all available features | 0.6088 | [0.5502; 0.6633] | +0.0296 | [−0.0280; +0.0856] | 0.309 |
| all minus `m3.p_faith` | 0.5663 | [0.5106; 0.6221] | −0.0128 | [−0.0663; +0.0413] | 0.652 |

Diagnostics of the winning run: `null_percentile` 100.0, `above_noise` true,
`axes.roc_auc` 0.6128 [0.5371; 0.6845], per-repeat F1
{0.6361, 0.6440, 0.6387, 0.6466, 0.6623}, `sd_across_repeats` 0.0092.

**Mandatory caveat.** `n_evaluated = 331`, `n_excluded_by_folds = 115`. The cohort is
the intersection of source coverages (the judge run covers 446 cases; folds leave 331),
i.e. 14.8% of the corpus. 0.6543 must never be printed next to 0.5339 as if they were
comparable. What *is* comparable is the paired delta on the same cohort: +0.0752.

Also worth reporting: the two `m3` axes behave in opposite directions — `p_faith`
is the only significant contributor in the project, while `p_rel` contributes
−0.0115. This is the same axis asymmetry the diagnosis predicted (AUC 0.627 vs 0.497),
and it is the strongest argument for the split-axes hypothesis.

---

## 5. Suggested article structure (what goes where)

Following the template's skeleton:

- **Introduction** — same two axes and Eq. (1). Add: both methods are constrained to
  run locally (banking data), which is why the judge is a 7B open model and Method 6
  uses a 280M-parameter cross-encoder.
- **Related Work** — LettuceDetect/MiniCheck stay. Add for Method 3: LLM-as-judge and
  logprob-based verdict scoring; GEPA/DSPy for prompt optimization; self-consistency.
  For Method 6: SelfCheckGPT (and *why it is inapplicable here* — this is a
  contribution, not a citation), plus NLI-based grounding.
- **Data** — replace the template's split section with §2 above. The leakage argument
  (24.9%, 1-NN at 0.6278) and the 753-case group exclusion are the two facts a reviewer
  will ask about.
- **Methods** — §3.2–3.4 for Method 6, §4.1–4.3 for Method 3. The logprob extraction
  and the two-class softmax are the two mechanisms that need equations.
- **Results** — §3.5–3.7 and §4.5. Keep cohort sizes in every table.
- **Discussion** — the label contingency (§2.1), the axis asymmetry, the pre-registered
  decision rule, and cost (§3.1).
- **Appendix** — feature distributions, the NLI ablation, per-repeat F1, prompts.

### Contribution list for Methods 3 and 6

1. A group-aware evaluation protocol for this corpus, with an explicit demonstration
   that the previous stratified split leaked 24.9% of questions and let a 1-NN
   memorizer beat every real method.
2. A logprob-based LLM judge whose continuous score is owned by the protocol, with a
   measured diagnosis of its failure modes (relevance-axis collapse, probability
   saturation, inverse context-length dependence).
3. A cost-driven reformulation of the SelfCheckGPT branch into direct NLI grounding —
   an ~8× reduction in NLI pairs and from ~37 GPU-hours to ~3 minutes — with a
   pre-registered decision rule and a resulting reproducible negative result.
4. Evidence that a judge's faithfulness probability is the only feature in this project
   with a statistically significant stacking gain (+0.0752, p = 0.013), while its
   relevance probability contributes nothing.

---

## 6. LaTeX-ready tables

```latex
\begin{table}[H]
\centering
\caption{Corpus and split. The group-aware split excludes one 753-case group,
so cross-validated metrics use 1,480 rows while feature-level ROC-AUC uses all 2,233.}
\label{tab:corpus}
\begin{tabular}{lr}
\toprule
Property & Value \\
\midrule
Cases                        & 2,233 \\
Faithful                     & 1,642 (73.5\%) \\
Relevant                     & 1,961 (87.8\%) \\
Reliable                     & 1,616 (72.4\%) \\
Faithful but irrelevant      & 26 \\
Unfaithful but relevant      & 345 \\
Groups after grouping        & 715 \\
Excluded (largest group)     & 753 \\
Cross-validated cohort       & 1,480 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Row-level results under the group-aware 5$\times$5 CV protocol.
Thresholds are fitted inside each fold's training part; intervals are percentile
bootstrap with $B=10{,}000$. The stacking row uses a different cohort and is not
comparable with the rows above it.}
\label{tab:row-level}
\footnotesize
\begin{tabularx}{\linewidth}{Xrlcr}
\toprule
Configuration & $\fmacro$ & 95\% CI & Above noise & $n$ \\
\midrule
Constant ``always reliable''      & 0.4203 & ---              & ---  & 1,480 \\
Independent rules                 & 0.5122 & [0.4866; 0.5382] & no   & 1,480 \\
Surface baseline                  & 0.5350 & [0.5102; 0.5615] & yes  & 1,480 \\
Method 6, \texttt{min\_entail}    & 0.5339 & [0.5070; 0.5601] & yes  & 1,480 \\
Method 6, ruModernBERT            & 0.5397 & [0.5128; 0.5658] & yes  & 1,480 \\
\midrule
Surface (stacking cohort)         & 0.5791 & [0.5222; 0.6348] & ---  & 331 \\
Surface $+$ \texttt{m3.p\_faith}  & 0.6543 & [0.5956; 0.7097] & yes  & 331 \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[H]
\centering
\caption{Method 6: ROC-AUC of every feature against faithfulness, $n=2{,}233$,
$B=10{,}000$. No interval departs from 0.5 by more than 0.06.}
\label{tab:m6-auc}
\footnotesize
\begin{tabular}{llr}
\toprule
Feature & ROC-AUC [95\% CI] & vs reliable \\
\midrule
\texttt{m6.mean\_entail}      & 0.556 [0.529; 0.583] & 0.554 \\
\texttt{m6.min\_entail}       & 0.548 [0.522; 0.575] & 0.547 \\
\texttt{m6.max\_entail}       & 0.525 [0.498; 0.552] & 0.520 \\
\texttt{m6.chunk\_spread}     & 0.510 [0.485; 0.535] & 0.507 \\
\texttt{m6.n\_sentences}      & 0.507 [0.481; 0.533] & 0.502 \\
\texttt{m6.cond\_coverage}    & 0.506 [0.479; 0.532] & 0.503 \\
\texttt{m6.max\_contra}       & 0.499 [0.473; 0.526] & 0.496 \\
\texttt{m6.n\_conditions}     & 0.494 [0.467; 0.521] & 0.491 \\
\texttt{m6.mean\_contra}      & 0.491 [0.464; 0.518] & 0.493 \\
\texttt{m6.digit\_coverage}   & 0.483 [0.456; 0.510] & 0.482 \\
\texttt{m6.uncovered\_max}    & 0.473 [0.450; 0.495] & 0.475 \\
\texttt{m6.frac\_unsupported} & 0.461 [0.440; 0.483] & 0.465 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Paired stacking ablation on the 331-case cohort. Deltas are candidate
minus the surface baseline on identical resampled indices.}
\label{tab:stack-ablation}
\footnotesize
\begin{tabular}{lrlrl r}
\toprule
Added feature & $\fmacro$ & 95\% CI & $\Delta$ & $\Delta$ 95\% CI & $p$ \\
\midrule
--- (surface base)             & 0.5791 & [0.5222; 0.6348] & ---     & ---                  & --- \\
\texttt{m3.p\_faith}           & 0.6543 & [0.5956; 0.7097] & $+$0.0752 & [$+$0.0158; $+$0.1344] & 0.013 \\
\texttt{m3.p\_rel}             & 0.5676 & [0.5108; 0.6228] & $-$0.0115 & [$-$0.0344; $+$0.0102] & 0.338 \\
\texttt{m6.min\_entail}        & 0.5686 & [0.5128; 0.6256] & $-$0.0105 & [$-$0.0547; $+$0.0348] & 0.657 \\
\texttt{m6.mean\_entail}       & 0.5678 & [0.5122; 0.6230] & $-$0.0114 & [$-$0.0528; $+$0.0317] & 0.617 \\
\texttt{m6.frac\_unsupported}  & 0.5654 & [0.5097; 0.6209] & $-$0.0137 & [$-$0.0537; $+$0.0273] & 0.524 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Cost of the two Method 6 formulations over the full corpus.}
\label{tab:m6-cost}
\begin{tabular}{lrr}
\toprule
 & SelfCheckGPT & NLI grounding \\
\midrule
NLI pairs        & $\approx$404,000 & 51,763 \\
Forward windows  & ---              & 69,478 \\
LLM generations  & 5--20 per case   & 0 \\
Wall clock       & $\approx$37 GPU-h & $\approx$3 min \\
\bottomrule
\end{tabular}
\end{table}
```

Equations to add:

```latex
% Two-class NLI normalization (neutral discarded)
\begin{equation}
 e_{ij} = \frac{\exp(z^{\mathrm{ent}}_{ij})}
               {\exp(z^{\mathrm{ent}}_{ij}) + \exp(z^{\mathrm{con}}_{ij})}
 \label{eq:two-class}
\end{equation}

% Weakest-link grounding
\begin{equation}
 q_j = \max_i e_{ij}, \qquad
 s_{\min} = \min_j q_j, \qquad
 s_{\mathrm{mean}} = \frac{1}{m}\sum_{j=1}^{m} q_j
 \label{eq:grounding}
\end{equation}

% Verdict probability from logprobs
\begin{equation}
 P(\mathrm{PASS}) =
 \begin{cases}
   \dfrac{e^{\ell_{\mathrm{PASS}}}}{e^{\ell_{\mathrm{PASS}}}+e^{\ell_{\mathrm{FAIL}}}},
     & \text{both sides in top-}k,\\[2ex]
   e^{\ell_{\mathrm{PASS}}}, & \text{only PASS visible},\\[1ex]
   1-e^{\ell_{\mathrm{FAIL}}}, & \text{only FAIL visible},\\[1ex]
   0.5, & \text{neither visible}.
 \end{cases}
 \label{eq:verdict-prob}
\end{equation}

% Self-consistency: average probabilities, not votes
\begin{equation}
 p = \frac{1}{N}\sum_{t=1}^{N} p_t, \qquad
 p_{\mathrm{std}} = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(p_t-p)^2}
 \label{eq:selfconsistency}
\end{equation}
```

---

## 7. Reproduction commands (for the "Code and artifacts" section)

```bash
# Method 6, whole corpus (~3 min on A100, ~10 min on T4, runs on CPU)
python scripts/score_m6_grounding.py --data data/alfa.jsonl \
  --output predictions/alfa/m6_grounding/base/scores.jsonl \
  --nli-model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 \
  --batch-size 64 --max-length 512 --overlap 128 --entail-threshold 0.5

python scripts/score_m6_grounding.py --auc-only --data data/alfa.jsonl \
  --scores predictions/alfa/m6_grounding/base/scores.jsonl \
  --auc-report predictions/alfa/m6_grounding/base/auc.json --bootstrap-B 10000

# Method 3, split axes (~20 min on A100 with vLLM up)
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --max-logprobs 25      # --max-logprobs is mandatory

python scripts/run_m3.py --data data/alfa.jsonl \
  --output predictions/alfa/m3_judge/axes/scores.jsonl \
  --mode zero_shot --backend openai_judge --prompt-style axes \
  --model Qwen/Qwen2.5-7B-Instruct --api-base http://localhost:8000/v1 \
  --concurrency 16

# Evaluation, identical for every method
python scripts/evaluate_cv.py --data data/alfa.jsonl \
  --folds data/splits/folds_alfa.json \
  --scores <scores.jsonl> --score-expr "m3.p_faith * m3.p_rel" \
  --compare predictions/alfa/baselines/surface/scores.jsonl \
  --output <report.json>
```

Both methods are also reproduced cell by cell in
`notebooks/standalone/method3_judge.ipynb` and
`notebooks/standalone/method6_grounding.ipynb`.

---

## 8. What is NOT measured — do not let the draft imply otherwise

| Claim the draft might be tempted to make | Actual status |
|---|---|
| "Split axes fixes the relevance axis" | **Not run.** Implemented (`m3/axes.py`, prompt `relevance/v2` without chunks), target AUC(`p_rel`) ≥ 0.60 vs current 0.497. No number exists. |
| "Self-consistency removes saturation" | **Not run.** Implemented (`m3/selfconsistency.py`), target share of `p_faith>0.99` < 30% vs current 50–79%. |
| "Per-chunk verification detects fact mixup" | **Not run.** Implemented (`m3/perchunk.py`), target recall(unreliable) ≥ 0.45 at precision ≥ 0.55. Only 18 corpus cases carry `reason_chunk_fact_mixup`, so this test is underpowered even once run. |
| "Marker feedback helps prompt optimization (H5)" | **Not established.** The earlier run was stopped by a pre-registered rule; post hoc the difference was indistinguishable from zero, CI [−0.037; +0.100]. The hypothesis remains *untested*, not refuted. Re-run with balanced accuracy is implemented, not executed. |
| "A fine-tuned judge beats a prompt-optimized one (H1)" | **Not run.** `methods/ft_judge/` and five DataSphere job configs exist; no checkpoint, no scores. |
| "Method 3 achieves 0.6543" | **Wrong framing.** That is a *stack* (surface + `m3.p_faith`) on a 331-case cohort. Method 3 alone under the CV protocol has no full-corpus number. |
| "Method 6 gives useful signal on faithfulness but not relevance" | **Overstated.** The axis difference lies inside the CIs (mean AUC 0.504 vs 0.499). Say the data do not resolve the asymmetry. |
| "Surface = 0.5350 and surface = 0.5791" | Both are correct — **different cohorts** (1,480 and 331). Always print $n$. |

Missing baselines that a reviewer may ask for: the encoder (RuModernBERT) and
LettuceDetect features were never scored under this protocol on this corpus, so the
present article cannot rank Methods 3 and 6 against them on equal footing. The
template's LettuceDetect numbers come from the *other* split (2,245 rows, 225-row test)
and must not be merged into the same table.

---

## 9. Provenance index

| Number | File |
|---|---|
| Corpus statistics, contingency, markers | computed from `data/alfa.jsonl` |
| Split stats, leakage, group sizes | `data/splits/folds_alfa.json` (`stats`) |
| M6 feature AUC + CI | `predictions/alfa/m6_grounding/{base,rumodernbert}/auc.json` |
| M6 row-level macro-F1 | `predictions/alfa/m6_grounding/*/report.json` |
| M6 pair and window counts, model, dtype | `predictions/alfa/m6_grounding/base/run.yaml` |
| Stacking table, cohort 331, diagnostics | `docs/report/wave3.md` §3.1 |
| `independent` baseline | `predictions/alfa/baselines/independent/report.json` |
| Judge diagnosis (AUC 0.627/0.497, saturation) | `docs/specs/30_PHASE2_метод3.md`, `docs/handoff/tasks/C3.md` |
| Legacy holdout numbers (n = 223) | `predictions/alfa/m3/*/report_test.json` |
| Runtime budgets | `docs/specs/90_DATASPHERE_runbook.md` §5 |
| Hyperparameters of the FT judge | `jobs/ft_judge_fold0.yaml`, `scripts/train_ft_judge.py --help` |
| GEPA settings | `src/rag_reliability/methods/m3/gepa.py`, `jobs/gepa_*.yaml` |
