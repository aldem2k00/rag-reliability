# Presentation materials: Methods 3 and 6

Slide-by-slide material for the team deck
(`Assessing_the_Reliability_of_Responses_in_RAG_Systems`, 26 slides), restricted to
Methods 3 and 6. Every figure is traceable — see §9 of
[`article_materials_m3_m6.md`](article_materials_m3_m6.md) for the provenance index.

---

## 0. Read this first: what the current deck gets wrong

The existing deck predates three waves of work. Five statements on it are now false or
unsupported, and a reviewer who checks the repository will find that out.

| Slide | Says | Should say |
|---|---|---|
| 15 (M6) | "SelfCheckGPT, N = 10 samples per case", "contradiction is the workhorse Δ = +0.939", "semantic entropy inverts at N = 10 (−0.164) — alignment collapse", "awaiting GPU" | The branch was **stopped and redefined** as NLI grounding. Δ = +0.939 was measured on the *synthetic* corpus. The entropy inversion is **0.38 standard errors** and its sign flips with the clustering threshold — no signal at the available power. Nothing is awaiting GPU: the method is measured. |
| 12 (M3) | "100% extraction on all 2233 real cases" | True for the local 7B run (82 distinct probability values, 0 invalid). **Not** true for the 72B runs: two distinct values (0.9 / 0.1), i.e. regex fallback, no logprobs. |
| 6 | "train 1787 / val 223 / test 223 (seed 42)" | Protocol replaced: **5×5 CV over group-aware folds**, threshold fitted inside each fold's training part. The old split leaked 24.9% of questions. |
| 18, 21 | "15 methods / 7 families", "2,245 cases", "best 0.588" | **19 methods / 9 families**, **2,233** deduplicated cases. 0.588 was measured under the old protocol and is withdrawn. |
| 20, 21 | "Supervised encoder wins (0.59)" | Not comparable with anything current. Under the canonical protocol the only significant result is `surface + m3.p_faith` = **0.6543** on a 331-case cohort. |

Fixing slide 15 is the highest-value edit in the deck: it currently presents a
retracted claim as a headline finding.

---

## 1. Slide "Method 3: LLM-as-judge" — replacement content

**Title:** Method 3 · LLM-as-judge with logprob verdicts

**Left column — what it is**

- Frozen `Qwen2.5-7B-Instruct` served locally by vLLM; no weights trained in the
  prompt-optimization line.
- The verdict is a token, but the **score is a probability**: `P(PASS)` read from the
  logprobs at the verdict position, softmax over the PASS/FAIL pair.
- The method never binarizes. The threshold belongs to the evaluation protocol and is
  fitted inside each fold's training part.
- Two prompts, versioned in `configs/prompts/`, quoting the annotation rubric
  verbatim. The relevance prompt **receives no chunks** — relevance is a relation
  between dialogue and answer; ~600 prompt tokens instead of ~3,000.

**Right column — the diagnosis that drives everything else**

| | |
|---|---:|
| AUC(`p_faith`) vs faithfulness | 0.627 |
| **AUC(`p_rel`) vs relevance** | **0.497** |
| Says `RELEVANCE: PASS` | 98.6–100% (gold 84.9%) |
| `p_faith` > 0.99 | 54% zero-shot → **76% few-shot** |
| Recall of unreliable | 0.22–0.33 |
| `p_faith`, 8 chunks vs 5 | 0.857 vs 0.826 |

**The one line to say out loud:** more demonstrations made the judge *more confident,
not more correct* — few-shot pushed saturation from 54% to 76%.

**Bottom strip — the payoff**

`m3.p_faith` is the only feature in the whole project with a statistically significant
stacking gain: **+0.0752**, 95% CI [+0.0158; +0.1344], p = 0.013. Its sibling
`m3.p_rel` contributes −0.0115. The axis asymmetry in the diagnosis reappears exactly
in the stack.

**Speaker note.** The mandatory smoke test exists because of a real failure: vLLM must
run with `--max-logprobs 25` (the client requests `top_logprobs=20`), otherwise the
server returns verdicts without probabilities, every score becomes 0.5, and a 2,233-case
run *looks successful while carrying no signal*. An earlier bug also applied a sigmoid
to a logprob instead of exponentiating it — inverting monotonicity — and a green unit
test kept it alive across three branches.

---

## 2. Slide "Method 6" — full replacement (this slide must change)

**Title:** Method 6 · NLI grounding — from SelfCheckGPT to a measured negative result

**Block 1 — why the original formulation was stopped**

SelfCheckGPT requires re-sampling **the same generator**. The production bot's prompt
was never released, so samples came from a different model with an invented prompt.
The measurement answers "do two different models disagree?", not "did the bot
hallucinate?" — uninterpretable at any value. Three supporting reasons: the method is
designed for *context-free* settings while here context exists and is largely subsumed
by direct entailment; the cost was ~37 GPU-h for a feature with no demonstrated
contribution; and the branch's own headline numbers did not survive audit (see §0).

**Block 2 — what replaced it**

Premise = a retrieved chunk, hypothesis = a sentence of the answer. The generator is
removed from the loop entirely.

| | SelfCheckGPT | NLI grounding |
|---|---:|---:|
| NLI pairs | ~404,000 | **51,763** |
| LLM generations | 5–20 per case | **0** |
| Wall clock | ~37 GPU-h | **~3 min** |

Twelve features: 8 grounding + 4 coverage, the latter reusing the same matrix with
zero extra NLI calls. Target feature `min_entail` = the **weakest link**: one
unsupported sentence makes the answer unfaithful, and a mean would wash it out.

**Block 3 — the result, and the rule that was fixed in advance**

Best feature ROC-AUC = **0.548 [0.522; 0.575]** (n = 2,233, B = 10,000). Swapping the
NLI model changes nothing (0.556 vs 0.557). In the stack all three tested features
give a **negative** delta.

Pre-registered decision rule (task card C4, written before the run):
≥ 0.60 → develop · 0.53–0.60 → features to the stack · ≤ 0.53 → close.
Observed 0.548 → middle band → **branch closed as a development direction.**

**The one line to say out loud:** this is a negative result we can defend — the method
is implemented in full, measured under the shared protocol, made 12× cheaper, and
reproduces in 15 minutes.

---

## 3. Slide "Hypotheses" — honest status of H4 and H5

Replace the current status wording with:

| | Hypothesis | Status |
|---|---|---|
| **H4** | SelfCheckGPT successors give useful faithfulness signal but underperform on relevance | **Partially answered, weaker than stated.** Signal on faithfulness is distinguishable from chance for only 2 of 12 features (AUC 0.548–0.556); the difference *between axes* lies inside the CIs (mean 0.504 vs 0.499). The data do not resolve the asymmetry. |
| **H5** | Markers as textual feedback strengthen prompt optimization | **Untested, not refuted.** The earlier run was stopped by a pre-registered rule; post hoc the difference was indistinguishable from zero, CI [−0.037; +0.100]. Re-run with balanced accuracy is implemented, not executed. |
| **H1** | Fine-tuned judge beats prompt-optimized judge | **Not run.** LoRA pipeline and five job configs exist; no checkpoint. |

The distinction between "untested" and "refuted" is worth a sentence on the slide.
It is the difference between an experiment that failed and an experiment that never
ran, and conflating them is how a team ends up believing a false negative.

---

## 4. New slide worth adding: "The split was leaking"

This is the most defensible methodological contribution and the deck does not mention
it at all.

- The previous split was stratified: **the same question landed in both train and
  test**. Measured leakage: 24.9% by question.
- Consequence: a 1-nearest-neighbour memorizer scored **0.6278** — higher than every
  real method on the leaderboard at the time.
- Fix: group-aware folds. Union-find over three edge sources — identical normalized
  last client utterance; cosine ≥ 0.9 over char 3–5-gram TF-IDF of the dialogue;
  identical first retrieved chunk. Group keys are content-addressed, so the assignment
  is stable across runs.
- Result: 715 groups, leakage 0.0 on all three checks.
- Price: one group holds **753 ids** (33.7% of the corpus — the same question asked
  over and over) and cannot sit in any fold without dominating it, so it is excluded.
  Every CV number uses **n = 1,480**; feature-level ROC-AUC, which needs no split, uses
  all 2,233.

**The one line:** we lost a third of the corpus and about 0.1 macro-F1 of apparent
performance, and got numbers that mean something.

---

## 5. Results slide — the only table that should be shown

| Configuration | macro-F1 | 95% CI | Above noise | n |
|---|---:|---|---|---:|
| constant "always reliable" | 0.4203 | — | — | 1,480 |
| `independent` (rules) | 0.5122 | [0.4866; 0.5382] | no | 1,480 |
| `surface` (text baseline) | 0.5350 | [0.5102; 0.5615] | yes | 1,480 |
| **Method 6**, `min_entail` | 0.5339 | [0.5070; 0.5601] | yes | 1,480 |
| Method 6, ruModernBERT | 0.5397 | [0.5128; 0.5658] | yes | 1,480 |
| surface (stacking cohort) | 0.5791 | [0.5222; 0.6348] | — | **331** |
| **surface + `m3.p_faith`** | **0.6543** | [0.5956; 0.7097] | yes | **331** |

Two rules for presenting it:

1. **Always print n.** The last two rows use a different cohort (the intersection of
   source coverages) and are not comparable with the rows above them.
2. **Never compare point estimates that differ by less than ~0.06** — that is the CI
   width on this corpus. Compare paired deltas instead.

Anticipated question — "why is your best number lower than the 0.588 on the old
deck?" Answer: 0.588 was measured on a leaking split under a different protocol and is
withdrawn. The honest comparison is against the floor, 0.4203, and against the surface
baseline on the identical cohort.

---

## 6. Numbers cheat-sheet (for Q&A)

**Corpus.** 2,233 cases · faithful 73.5% · relevant 87.8% · reliable 72.4% ·
5–8 chunks (median 5) · answer median 418 chars · context median 6,410 chars ·
4.05 sentences per answer, 9,053 in total.

**Contingency.** faithful∧irrelevant = 26 · unfaithful∧relevant = 345 ·
both = 1,616 · neither = 246. So relevance can rescue at most 26 unreliable cases,
while every relevance false negative on 1,616 reliable rows damages the conjunction.

**Markers.** 72.4% `none`, **21.4% `unknown`**, and 13 codes over the remaining ~6%.
The largest error class is `reason_hallucinated_fact` at 32 cases (1.4%);
`chunk_fact_mixup`, the target of per-chunk verification, has **18**. Per-marker claims
are underpowered by construction.

**Method 6 cost.** 51,763 NLI pairs, 69,478 forward windows, fp16 on CUDA, batch 64,
`max_length` 512, overlap 128, ~3 min on A100 and ~10 min on T4 — it even runs on CPU.

**Method 3 budgets (A100, 2,233 cases).** holistic ~15 min (6.7M prompt / 0.7M
completion tokens) · relevance axis alone ~5 min (1.3M / 0.4M) · self-consistency k=8
~50 min (prefix-cached prompt, 5.4M completion) · per-chunk ×8 ~40 min · GEPA ~1.2 h
per run, ~7 h for the full H5 design · FT judge ~1.5 h per fold, ~7.5 h for five.

**Protocol.** 5×5 CV · threshold grid step 0.01, smallest maximizer wins · bootstrap
B = 10,000 · 500 null runs · a report without a CI fails schema validation.

**Engineering.** 19 methods in one registry · 835 tests · every run writes `run.yaml`
with config, git hash, dirty flag and seed · judge calls cached, so any run resumes.

---

## 7. Demo suggestions

Three things that show well in two minutes each.

1. **The NLI matrix on one case.** `notebooks/standalone/method6_grounding.ipynb`
   prints the sentences × chunks entailment matrix, then shows `min_entail` as the
   minimum over per-sentence maxima. The method becomes obvious in one screen, and the
   notebook asserts its own formulas against `methods/m6/grounding.py` so what is shown
   is what was computed.
2. **The logprob smoke test.** Five cases, two assertions; then show what a broken run
   looks like — the 72B artifact with two distinct values across 223 cases.
3. **`rag-judge list-methods`** — 19 methods with their requirements, straight from the
   registry that also drives the CLI, the benchmark and the demo.

---

## 8. Three slide-worthy diagrams to draw

1. **Method 6 pipeline.** answer → `razdel` → sentences; context → chunks; then an
   m × n matrix of entailment probabilities; `max` along the row → per-sentence
   support; `min` over rows → `min_entail`. Annotate the two-class softmax
   (neutral discarded) and premise windowing as callouts.
2. **Method 3 verdict extraction.** the three output lines → locate the `AXIS:` anchor
   → the verdict token → the top-logprobs list with PASS and FAIL highlighted →
   softmax over the pair → `p_faith`. Add a red branch: "no `--max-logprobs` → no
   logprobs → p = 0.5 for every case".
3. **Cohorts.** 2,233 total → 1,480 in folds (753 excluded as one group) → 331 in the
   stacking cohort. This single picture prevents most of the misreadings of the results
   table.
