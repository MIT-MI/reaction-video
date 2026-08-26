# ReactionBench gold-eval — Experiments evidence record (2026-08-19)

Researcher-facing evidence base for the Experiments section. NOT paper prose; NOT final
artifacts. Every number here traces to a file in this repo (provenance ledger §2); numbers
that cannot be traced are marked NOT-FINAL. Slices are kept separate — no silent merging.

---

## 1. Experiment index

| ID | Question | Items | Models | Status |
|---|---|---|---|---|
| T2-MCQ | Can models pick the triggering same-video stimulus segment (single-call, 4-option)? | 150 | 7 arms | complete |
| T1-JOINT | Can models rank a video's moments by reaction intensity, seen jointly? | 250 videos / 642 moments | 7 arms | complete (35B: 237/250) |
| T3-RAT | Can models write the human-matching rationale (blind, judged cross-family)? | 200 | 7 arms | complete (gemini gen: 190/200 usable) |
| E1-VIDEO | Frames (1fps, 512px) vs native mp4 (incl. audio) — same model, same items | T2+T1 | Gemini 3.7 Flash | complete |
| E2-SCORED | Per-option 1-10 scoring (old-paper protocol) vs single-call MCQ | 150×4 calls | 7 arms | complete |
| E3-INDEP | Independent per-moment 0-100 scoring (no cross-moment context) vs T1-JOINT | 250 videos | 7 arms | complete w/ coverage gaps (172-241/250) |
| V3-SBS | Side-by-side (reaction\|option) native video, per-option scoring | 150×4 | Gemini 3.7 Flash | complete |
| HC-RANK | Human LOO ceiling on the ranking set | 420 rater-video pairs | — | complete |
| JJ-AGREE | Judge-judge agreement on 30-item overlaps | 60 rows/model ×5 | GPT-5 vs Gemini judges | complete |

Roster decisions: Claude dropped (owner 2026-08-11); Gemini Pro & GPT-4o dropped as stale
(owner 2026-08-19; ADR-0003). Fine-tune arm: designed (ADR-0003), not yet run.

## 2. Provenance ledger

Repos/commits: eval = `MIT-MI/reaction-video@gold-eval` (results as of `77e27cb`);
dataset = `reaction_video_benchmark_phase0_pilot@phase0/pilot` (`b1072ec`).

| Evidence | File (eval repo unless noted) |
|---|---|
| Master aggregate | `gold_eval/results/summary.json` (regenerate: `python -m gold_eval.summarize --with_ceiling`) |
| T2 predictions | `gold_eval/results/match/<model>.jsonl` (`__video` = E1 arm) |
| E2 / V3 predictions | `gold_eval/results/match_scored/`, `gold_eval/results/match_sbs/` |
| T1 predictions | `gold_eval/results/ranking/` (joint; `__video` = E1), `gold_eval/results/ranking_indep/` (E3) |
| T3 generations / judgements | `gold_eval/results/rationale/`, `gold_eval/results/rationale_judge/<gen>__by__<judge>.jsonl` |
| Tie-break slices (E2/V3) | computed 2026-08-19 from the scored jsonls (see §3.3; recompute snippet in git history) |
| Task sets (frozen) | `gold_eval/tasks/{match_set,ranking_set,rationale_set,rationale_refs}.json` |
| Ranking-set construction | dataset repo `scripts/pilot/build_ranking_tasks.py`, `reports/phase0/ranking/selection_record.json` |
| Match-set construction | dataset repo `tools/data_annotation/gold_platform/build_match_tasks.py`, Supabase `gold_match_tasks` |
| Human rationales (Prolific) | Supabase `gold_rationales`; snapshot in `gold_eval/tasks/rationale_refs.json` (447 'ok' refs) |
| Stage A agreement stats | dataset repo `docs/gold_stage_a_figures/stats.json` (α_ordinal=0.357), write-up `docs/GOLD_STAGE_A_ANALYSIS.md` |
| Spend/token ledger | `gold_eval/results/costs/*.jsonl`; `python -m gold_eval.costs` |
| Prompts (exact strings) | `gold_eval/run_{match,ranking,rationale,match_scored,ranking_independent,match_sbs}.py`, `gold_eval/judge_rationale.py` |
| Run logs | `gold_eval/data/*.log` (p2/p3/p4/p5 batches) |

Input protocol: frames 1 fps, max side 512, caps: match reaction 12 / option 6, ranking 64,
rationale 12/stream, E3 window 12. Native-video arm sends mp4 bytes incl. audio (oversized
clips re-encoded, `frames.video_for_upload`). Temperature 0 (GPT-5: reasoning_effort=minimal;
Gemini: thinking_level=low + headroom).

## 3. Canonical result tables

### 3.1 T2 match — single-call MCQ, REBUILT set (112 items, adaptive pairwise gap 10/8/6 s; chance .25)

Rebuilt 2026-08-24 (C12 fix); pre-rebuild predictions archived in `results/match_pre_rebuild/`
and their numbers in git history. Ground truth = temporal alignment; G9 human confirmation
pass pending (owner, ~2-3.5 h) — it supplies the human reference, not a validity gate.

| Arm | acc | 95% CI | unparsed | err time-dist med (s) |
|---|---|---|---|---|
| Gemini 3.7 Flash native-video | **.545** | [.453, .634] | 0 | 27.5 |
| GPT-5 (frames) | .509 | [.418, .600] | 0 | 26.8 |
| Gemini 3.7 Flash (frames) | .500 | [.409, .591] | 1 | 24.8 |
| Qwen3.6-35B-A3B | .482 | [.392, .574] | 0 | 26.8 |
| Qwen3.6-27B | .446 | [.358, .539] | 0 | 25.7 |
| Qwen3.5-397B-A17B | .411 | [.324, .503] | 0 | 24.7 |
| Kimi-K2.6 | .375 | [.291, .467] | 0 | 24.5 |
| Qwen3.5-9B | .259 (=chance) | [.187, .347] | 20 | 26.4 |

vs the pre-rebuild (contaminated) set: absolute accuracies moved little overall (top arms
+2-5 pt — cleaner distractor sets, no near-dup pairs shrinking effective option count), but
mid-field ORDER changed (35B .367→.482, Kimi .460→.375): per-model numbers on the old set
were not trustworthy, the rebuild mattered. E1 on the rebuilt set: video .545 vs frames .500,
McNemar p=.54 (n.s., direction consistent again; 24 video-only vs 19 frames-only correct). Error temporal distance grew (~19→~26 s
median) as expected — options are farther apart by construction. Cultural strata remain
descriptive only (C7).
### 3.2 T1 ranking — joint vs independent (250 videos, 642 moments; human LOO ceiling ρ=.649, 420 rater-video pairs)

| Arm | JOINT mean ρ | JOINT pooled r | JOINT n | INDEP mean ρ | INDEP pooled r | INDEP n |
|---|---|---|---|---|---|---|
| Gemini 3.7 Flash native-video | .2604 | .417 | 250 | — | — | — |
| Gemini 3.7 Flash (frames) | .1755 | .308 | 249 | .3282 | .5175 | 241 |
| Qwen3.6-27B | .1634 | .238 | 250 | **.3937** | .434 | 209 |
| Kimi-K2.6 | .0864 | .241 | 250 | .2856 | .430 | 204 |
| Qwen3.5-397B-A17B | .0646 | .217 | 250 | .3309 | .387 | 182 |
| Qwen3.6-35B-A3B | .0456 | .187 | 237 | .2566 | .398 | 205 |
| Qwen3.5-9B | .0294 | .204 | 250 | .3340 | .389 | 191 |
| GPT-5 | .0262 | .293 | 250 | .2039 | .426 | 217 |

mean ρ = per-video Spearman vs Stage-A consensus means, averaged over videos with non-tied
gold; pooled r = Pearson over all moments pooled. INDEP coverage gaps: runner drops a video
if any moment's answer is unparsable (caveat C3). Stage-A full-pool reference: consensus-mean
human ceiling r≈.51, within-video ρ≈.40 (dataset repo GOLD_STAGE_A_ANALYSIS.md — different
pool than the 250-video eval set; do not mix).

### 3.3 E2 per-option scoring & V3 side-by-side (150 items × 4 calls; three scoring conventions — keep all three)

| Arm | strict acc (tie=wrong) | tie rate | expected acc (random tie-break) | decided-only acc (n decided) |
|---|---|---|---|---|
| GPT-5 | .200 | .520 | .378 | .417 (72) |
| Qwen3.6-35B-A3B | .207 | .667 | .392 | .620 (50) |
| Kimi-K2.6 | .187 | .667 | .387 | .560 (50) |
| Gemini 3.7 Flash | .147 | .627 | .349 | .393 (56) |
| Qwen3.5-397B | .140 | .647 | .303 | .396 (53) |
| Qwen3.5-9B | .133 | .733 | .326 | .500 (40) |
| Qwen3.6-27B | .093 | .813 | .307 | .500 (28) |
| **V3 SBS (Gemini)** | .147 | .613 | .322 | .379 (58) |

No item had all four options unparsable. Decided-only subsets are self-selected (easier
items) — NOT comparable to MCQ accuracy (caveat C4).

### 3.4 T3 rationale — cross-family judge (200 items; refs/item: 1×41, 2×77, 3×78, 4-5×4)

Primary judge GPT-5 (all non-OpenAI gens), Gemini 3.7 Flash judges GPT-5's gens:

| Generator | judge | cue (0-2) | link (0-2) | faithful (0-1) | composite (0-5) | n |
|---|---|---|---|---|---|---|
| GPT-5 | gemini | 1.470 | 1.335 | .870 | **3.675** | 200 |
| Kimi-K2.6 | gpt-5 | .935 | 1.495 | .185 | 2.615 | 200 |
| Qwen3.5-397B | gpt-5 | .745 | 1.475 | .180 | 2.400 | 200 |
| Gemini 3.7 Flash | gpt-5 | .738 | 1.084 | .534 | 2.356 | 191 |
| Qwen3.6-35B-A3B | gpt-5 | .750 | 1.420 | .185 | 2.355 | 200 |
| Qwen3.6-27B | gpt-5 | .720 | 1.380 | .180 | 2.280 | 200 |
| Qwen3.5-9B | gpt-5 | .675 | 1.410 | .135 | 2.220 | 200 |

Overlap slices (same 60 gens judged by BOTH judges; 5 Tinker models): Gemini-judge composite
runs 3.43-3.57 vs GPT-5-judge 2.22-2.62 on identical generations → **Gemini judge is
systematically ~+1.1-1.4 more lenient**; rank agreement Pearson .76-.85. Therefore the GPT-5
row (3.675, Gemini-judged) is NOT column-comparable to the others (caveat C1).
Qwen3.5-9B row doubles as the draft-model control (drafts were never shown to annotators).

### 3.5 Paired tests & confidence intervals (G1/G2, added 2026-08-19)

**G1 — joint vs independent, per-model video intersections** (`summary.json → ranking_paired`):

| Model | n paired | joint ρ | indep ρ | Δ mean | Δ 95% CI (boot) | Wilcoxon p |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 191 | .017 | .334 | +.317 | [.149, .480] | .00016 |
| Qwen3.5-397B | 182 | .082 | .331 | +.249 | [.099, .400] | .0024 |
| Qwen3.6-27B | 209 | .168 | .394 | +.226 | [.081, .373] | .0015 |
| Qwen3.6-35B-A3B | 196 | .036 | .243 | +.207 | [.041, .374] | .0126 |
| Kimi-K2.6 | 204 | .098 | .286 | +.188 | [.046, .329] | .0076 |
| GPT-5 | 217 | .035 | .204 | +.169 | [.021, .312] | .0300 |
| Gemini 3.7 Flash | 240 | .178 | .326 | +.148 | [.032, .268] | .0140 |

All 7 deltas positive, all CIs exclude 0 → the E3 mechanism finding is FINAL (C3 mitigated).

**E1 paired tests (same model, same items):** match McNemar on 149 shared items:
video-only-correct 33 vs frames-only-correct 25, p = .358 (n.s.). Ranking Wilcoxon on 249
shared videos: frames .176 → video .258, Δ = +.083, CI [-.049, .213], p = .271 (n.s.).
→ E1 direction is consistently positive on both tasks but NOT significant at current n;
the solid E1 result is the deterministic ~30× input-token reduction (ledger).

**G2 — 95% CIs** now attached to every entry in `summary.json` (`acc_ci95_wilson`,
`mean_spearman_ci95_boot`, `composite_ci95_boot`; bootstrap 10k, seed 0). Notables:
GPT-5 match [.408, .566]; Gemini-video match [.437, .596]; GPT-5 joint ranking CI
[-.083, .135] — statistically indistinguishable from zero.

### 3.6 Spend / scale (ledger)

Gemini $44.7 (caps enforced), OpenAI $3.8, Tinker grant $66.6. ~26k logged calls.

## 4. Caveat register

- **C1 Judge asymmetry.** T3 primary scores come from two different judges by design (no
  self-family judging). Cross-judge offset ≈ +1.1-1.4 (Gemini lenient). Any cross-column
  comparison must be within-judge, or offset-adjusted with the overlap slices. GPT-5-gen
  offset-adjusted composite ≈ 2.3-2.6 (NOT-FINAL; derived, choose method before use).
- **C2 Judge-human validation missing.** Judge scores are not yet anchored to human ratings
  of model rationales (ADR-0001 requires it). 30-50-item human anchor set TODO.
- **C3 E3 coverage.** INDEP drops whole videos on any unparsable moment: coverage 182-241/250,
  differs BY MODEL → mean ρ comparisons across models carry selection noise. Per-video paired
  comparison (joint vs indep on the intersection) is the clean test; not yet computed.
- **C4 E2/V3 decided-only slices** are self-selected easy subsets; report only alongside tie
  rate and expected-acc; never as headline accuracy.
- **C5 Native-video arm includes audio** (full-modality arm). E1 gains conflate video-token
  representation + audio access. A muted-video ablation would separate them (not run).
- **C6 T2 ground truth is auto temporal alignment** (match-ready filter + min-duration/
  distinctness guards). The planned 1-click human confirmation study was not run; 1 known
  defective option clip (0.2s) documented, first-frame fallback used.
- **C7 Cultural strata are tiny** (n=22/8/120): stratum-level accuracies are descriptive only;
  no significance claims.
- **C8 Creator concentration.** Ranking set: 4 Patreon reviewers = 80% of videos (cap relaxed
  35→50 to reach 250; selection_record.json). Disclose; per-creator slices not yet computed.
- **C9 Single run per arm**, temperature 0; no seed variance. Binomial 95% CI on 150-item
  accuracy ≈ ±8pt; on 250-video mean ρ, bootstrap CIs not yet computed.
- **C10 Gemini rationale gens**: 10/200 empty (thinking-budget bug era; 9 unjudged) → n=191.
  Regenerable at trivial cost if a clean 200 is wanted.
- **C12 (2026-08-19, SUPERSEDES the T2 numbers in §3.1/§3.3) Match option sets are flawed:**
  the builder enforced distractor-vs-correct separation but NOT distractor-vs-distractor —
  98/150 tasks contain two options starting <4s apart (median min pairwise start-gap 2.5s),
  i.e. adjacent slices of the same scene, visually near-identical (found by the owner during
  the G9 human pass; audit in git history). Zero temporal IoU between options, and every
  distractor is ≥4s from the correct option, so answers remain well-defined — but on affected
  items the effective option count is <4 (chance between .25 and .33), so absolute T2
  accuracies are possibly inflated and human confirmability is degraded. Builder fixed
  (pairwise ≥10s, farthest-first); rebuild requires server source data (~3 days), then T2
  re-runs for all arms and the G9 pass restarts. RESOLVED 2026-08-24: §3.1 now reports the rebuilt set; §3.3 (E2/V3) still reflects the pre-rebuild options and is kept as design-justification evidence only.
- **C11 Stage-A ceilings vs eval-set ceilings differ by pool** (α=.357/r≈.51/ρ≈.40 on 3,257
  moments vs LOO ρ=.649 on the curated 250-video set). Always name the pool.

## 5. Table/figure decision log (+ analysis captions)

**A1. Main table — three-task leaderboard (T2 MCQ acc, T1 joint ρ, T3 composite) + human
reference rows.** Decision: MAIN TABLE. Rationale: the paper's central claim (tasks are
well-posed, discriminative, unsaturated) needs all three tasks against ceilings in one view.
T3 column must be within-judge (GPT-5-judged models) with the GPT-5-gen row footnoted (C1).

> analysis_caption: Main results on the ReactionBench gold set: stimulus→reaction matching
> (150 four-option items, accuracy, chance 0.25), within-video intensity ranking (250 videos,
> mean per-video Spearman ρ against the mean of ≥3 anchored human scores; human leave-one-out
> ceiling 0.649), and rationale generation (200 items, 0-5 rubric composite from a cross-family
> LLM judge scored against 1-5 independent human-written references). Rows are model arms under
> an identical 1 fps sampled-frame budget; higher is better in all columns. The headline
> pattern: matching accuracy orders cleanly with model capability (26%→52%) yet the best model
> stays ~35 points below perfect despite objective ground truth, and joint ranking collapses to
> near-zero correlation for every arm (≤0.26 vs human 0.649) — models detect reactions but
> cannot calibrate intensity within a video. Supports the claim that the redesigned same-video
> tasks are discriminative and far from saturation. Rationale scores are comparable only within
> the shared judge; the GPT-5 row uses a different (more lenient) judge and is marked. Single
> run per arm at temperature 0; 150-item accuracies carry ≈±8-point binomial 95% CIs. This
> table does not show input-modality effects (see the frames-vs-video comparison) and does not
> imply the ranking failure is perceptual (see the independent-scoring ablation).

**A2. E1 frames vs native video (Gemini, T2+T1).** Decision: DOWNGRADED 2026-08-19 →
APPENDIX TABLE + main-text prose. Paired tests are n.s. on both tasks (match McNemar p=.36,
ranking Wilcoxon p=.27, §3.5); the defensible claims are (i) direction consistent across
both tasks and (ii) ~30× cheaper input tokens (deterministic). A "video beats frames"
main-table claim would exceed the evidence.

> analysis_caption: Input-modality comparison on a single model (Gemini 3.7 Flash) evaluated
> twice on identical items: once with the benchmark's standard 1 fps / 512px sampled frames,
> once with the raw video file (audio track included). Native video is directionally better on
> both tasks — matching 46.0%→51.7% (149 paired items) and ranking mean Spearman 0.18→0.26
> (249 paired videos) — but neither paired difference reaches significance (McNemar p=0.36;
> Wilcoxon p=0.27), while the input-token cost drops ~30×, which is a deterministic
> measurement. Supports only the narrow claims that the frame protocol does not overstate
> model ability (difficulty is conservative in direction) and that native video is far
> cheaper at equal-or-better quality for this model. Readers should not conclude that native
> video significantly improves accuracy, that the trend generalizes beyond this one model, or
> that the effect is purely visual — the native file also carries the audio track.

**A3. E3 joint vs independent scoring (all 7 arms).** Decision: FIGURE (paired dot / slope
chart, one line per model, joint→indep mean ρ, human ceiling as reference line) + APPENDIX
TABLE with ns. Rationale: the reversal (every model improves without context) is the paper's
sharpest mechanism finding; shape beats exact values. STATUS UPDATE 2026-08-19: the paired-
intersection recompute is done (§3.5) — all 7 deltas positive, CIs exclude 0, Wilcoxon
p≤.03 — so the finding is FINAL; the figure should plot the PAIRED means from §3.5, not the
raw per-file means.

> analysis_caption: Within-video relative calibration, not absolute perception, is what
> breaks current models on intensity ranking. Each line is one model's mean per-video Spearman
> ρ against human consensus under two elicitations over the same 250-video set: JOINT (the full
> reaction video plus all of its moments in one prompt, scored together) versus INDEPENDENT
> (each moment's window scored 0-100 in isolation on the same behaviorally anchored scale).
> Every arm improves when moments are scored independently — e.g. GPT-5 rises from 0.03 to 0.20
> and Qwen3.6-27B from 0.16 to 0.39, approaching half the human leave-one-out ceiling of 0.649
> (dashed reference) — and the model ordering under independent scoring roughly follows
> capability while joint scores cluster near zero regardless of scale. Supports the narrow
> claim that long-context relative comparison is the binding failure mode, echoing (and
> cleanly re-measuring) the context ablation of the prior submission. Independent-scoring
> coverage varies by model (182-241 of 250 videos) because items with unparsable single-moment
> answers are dropped; treat cross-model gaps under ~0.05 as noise and rely on the
> paired-intersection version for the final figure. Do not read this as evidence that context
> hurts humans — the human ceiling is measured on the joint task.

**A4. E2 per-option scoring + V3 side-by-side (three scoring conventions).** Decision:
APPENDIX TABLE + one PROSE sentence in main text. Rationale: it justifies the single-call MCQ
design (the old protocol's scores are tie-saturated on same-video negatives) but three
conventions are too much nuance for the main text. Misleading-risk: reporting only strict
accuracy ("below chance") would be rhetorically strong but statistically unfair — expected
accuracy with random tie-break (0.30-0.39) must appear.

> analysis_caption: Why the benchmark presents all four candidates jointly: replicating the
> prior submission's per-option protocol (reaction + one candidate per call, 1-10 likelihood,
> argmax over four calls; 150 items × 4 calls per model) produces ties on 52-81% of items —
> models assign the same top score to multiple same-video segments — so strict accuracy
> (ties counted wrong, 0.09-0.21) collapses below the 0.25 chance floor while the fairer
> random-tie-break expectation sits at 0.30-0.39, and the tie-free subset (0.38-0.62 accuracy,
> n=28-72) is a self-selected easy cohort not comparable to the main matching accuracy. The
> side-by-side variant (reaction and candidate hstacked into one video, Gemini only) shows the
> same tie saturation (61%), indicating that independently presented same-video negatives are
> indistinguishable on an absolute scale regardless of layout. Supports the narrow claim that
> same-video hard negatives require joint presentation to be discriminative — the design
> answer to the prior review's negative-mining critique. Not evidence that these models are
> below chance at matching, nor that side-by-side composition helps or hurts beyond the tie
> effect.

**A5. T3 rationale rubric breakdown + judge-agreement overlap.** Decision: MAIN TABLE
(within-judge columns) + APPENDIX TABLE (overlap slices, both judges). Rationale: T3 is a
headline task; the judge-asymmetry handling must be visible to preempt the circularity
critique. Misleading-risk: mixing judges in one column (C1).

> analysis_caption: Rationale quality under a cross-family LLM-as-judge protocol: each
> model's one-sentence explanation of 200 gold moments is scored against 1-5 independent
> human-written references on cue identification (0-2), stimulus-reaction linkage (0-2), and
> faithfulness (0-1). No model family judges its own generations; the five open-weight arms
> and Gemini are scored by GPT-5 (directly comparable), GPT-5's generations by Gemini (shown
> separately — a 60-item dual-judged overlap per model shows the Gemini judge is uniformly
> ~1.1-1.4 points more lenient at rank agreement r≈0.76-0.85, so its absolute scores are not
> column-comparable). Within the shared judge, composites span only 2.22-2.62 of 5 with cue
> scores below 1.0 throughout: models produce causally-worded but under-specified rationales
> that rarely pin the same concrete trigger humans name — supporting the claim that grounded
> reaction explanation, not fluent affect talk, is the open gap. Judge scores are not yet
> validated against human ratings of model outputs; treat orderings within ±0.15 composite as
> unresolved. The Qwen3.5-9B row also serves as the disclosure control that the annotation-era
> draft model holds no advantage (it ranks last).

**A6. Cultural-specificity slices (T2).** Decision: PROSE-ONLY (one sentence, descriptive).
Rationale: strata n=22/8/120 (C7) cannot support a table or claims beyond "no visible
universal-vs-niche gap at current power".

**A7. Human ceilings block (α, r, ρ by pool).** Decision: reported inside A1/A3 captions +
one METHODS paragraph, not a separate artifact. Rationale: pool confusion risk (C11) is best
handled in text.

**A8. Cost/efficiency table (tokens & $ per arm).** Decision: APPENDIX TABLE. Rationale:
reproducibility service to readers; also carries the frames-vs-video 30× token result.

**A9. Failure-mode taxonomy of T3 rationales.** Decision: OMIT for now — no coded data yet
(old paper's taxonomy exists but applies to the old task; recoding on new outputs is an open
task, see gaps G6).

## 6. Unresolved evidence gaps

- ~~G1~~ RESOLVED 2026-08-19: paired intersections computed (§3.5); A3 final.
- ~~G2~~ RESOLVED 2026-08-19: Wilson/bootstrap CIs in `summary.json` (§3.5). Consequence:
  E1 accuracy gains are n.s. → A2 downgraded. Single-run-per-arm caveat (C9) remains.
- **G3 (missing study)** Judge-human anchor validation (C2). ~30-50 items × human rubric
  scores of model rationales. Blocking the T3 claim strength.
- **G4 (missing number)** Gemini rationale regeneration for the 10 empty items (C10) if a
  clean n=200 row is wanted.
- ~~G5~~ RESOLVED 2026-08-21: DeepSeek-V3.1 (neutral family) scored all 7 generators
  (`*__by__tinker_deepseek-ai_DeepSeek-V3.1.jsonl`). Composite 0–5 [95% CI]: Kimi 2.91
  [2.68,3.14], Qwen3.5-397B 2.91 [2.68,3.14], Qwen3.5-9B 2.83, GPT-5 2.78 [2.54,3.02],
  Qwen3.6-35B 2.78, Gemini 2.65 (n=191), Qwen3.6-27B 2.61 [2.37,2.85]. All CIs overlap: under
  one comparable judge, rationale quality does NOT separate models; cue identification is the
  hard dimension (≤1.25/2 for every model); GPT-5 has the highest cue (1.25) but the lowest
  link (1.07). This column replaces the mixed-judge comparison in the main table (PAPER_NOTES B4/C4).
- **G6 (missing analysis)** Failure-mode coding of new T3/T2 errors; per-creator and
  per-domain slices (C8); Gemini prescreen self-selection check (prescreen_rank vs its own
  ranking scores — data present in ranking_set.json, analysis not run).
- **G7 (claim risk)** "Native video is better" generalizes only to Gemini (C5, single model,
  audio confound). Keep the claim model-scoped.
- **G8 (missing arm)** Fine-tuned Qwen baseline (ADR-0003) — designed, not run. Rebuttal
  item #2 depends on it. Inkling audio arm also not run (optional).
- **G9 (instrument mismatch) — ESCALATED 2026-08-19**: the human confirmation pass STARTED and
  immediately surfaced C12 (near-duplicate distractors on 98/150 tasks). Sequence now:
  rebuild option sets on the server (builder already patched) → reload gold_match_tasks →
  re-run T2 for all arms (~$6 + grant) → redo the human confirmation. All current T2 numbers
  are PROVISIONAL until then.

## 7. Handoff summary (for paper-writing)

Safe-to-write now: (i) three-task main results with ceilings (A1) and every number in §3;
(ii) the E1 modality-effect claim scoped to Gemini; (iii) the E2 tie-saturation argument as
design justification; (iv) the E3 calibration-vs-perception mechanism as the central analysis,
flagged preliminary pending G1; (v) methods facts (protocol, human data: 3,257 moments ×
≥3 raters, α=.357; 447 Prolific rationale refs from 3 annotators at 200/200 coverage;
frozen sets 150/250/200). Not safe yet: cross-judge T3 comparisons (G5), CI-qualified
superiority claims (G2), any fine-tune or audio-arm statement (G8), human-confirmed T2 gold
(G9). Numbers must be quoted from §3 / summary.json, not recomputed ad hoc.

---
### Addendum 2026-08-21 (meeting decisions; fine-tune track)
- **C13 (revised 2026-08-21)** `annotator_id='test'` (58 of 14,894 Stage-A scores) is the
  first author's own careful pass, not a throwaway account — retained as an author rater
  (ADR-0002 permits author raters, blind to model scores) and disclosed as such. No set
  rebuild; frozen membership unchanged.
- **Ground-truth normalisation — DONE, null effect:** rater-debiased consensus (two-way
  additive model, 8 raters, effects −0.30…+0.21) added to the frozen 250-video set
  (`ranking_set.json → moments[].debiased`, `debias_record.json`). |debiased − mean| averages
  0.045 (max 0.057) and changes the within-video ORDER in **0/250 videos**, so every Spearman
  ρ is identical to 4 decimals under both ground truths (`summary.json →
  mean_spearman_debiased`). Reason: ≥3 raters per moment and the same raters covering a
  whole video make rater offsets cancel within-video. Report as a one-line robustness check.
- **A10 (planned) Fine-tune with/without row** — PLAN in `finetune/PLAN.md`: LoRA on
  Qwen3.5-9B, F-INDEP + F-JOINT formats, train 1,364 videos / 1,952 moments (video-disjoint
  from every eval set), eval via the local HF path for both base and LoRA. Pre-registered
  success: ΔρJOINT ≥ .10, paired Wilcoxon p<.05. Status: data built, code untested on node.
- **Workshop target:** IAEval (Evaluation of Interactive Agents), DDL 2026-08-29/30.

---
### Addendum 2026-08-22 — fine-tune run 1 (A10) and the tie-aware scoring convention

**A10 result (finetune/REPORT.md, finetune_paired.json):** Qwen3.5-9B + LoRA r16
(29.1M trainable, 264 steps / 3 epochs, 3.2 h on 8×A6000), base and LoRA both through the
local HF path (local base reproduces the Tinker zero-shot row: joint .024 vs .029, indep
.318 vs .334). Pre-registered paired test: T1-JOINT Δρ = **+0.105**, CI [−0.022, 0.231],
Wilcoxon p = **0.131**, n = 166 → **criterion NOT met** (effect-size bar cleared,
significance not). T1-INDEP Δρ = +0.184, p = .003, n = 136. Joint-vs-indep gap *widened*
(.29 → .39): supervised intensity labels did not repair within-video calibration. Report as
a null result for the calibration hypothesis at this scale (single seed, C9).

**C14 — scorer artefact (applies to ALL arms, found via A10):** Spearman is undefined when a
model emits identical scores for every moment, and the pre-registered scorer *drops* such
videos, silently shrinking that arm's denominator. The LoRA collapsed its output range
(pred SD 27.9 → 9.8; 84/250 joint videos all-tied vs 0 for base), so its headline means
are flattered; base models also tie on INDEP (e.g. base Qwen 51/250). A tie-aware
convention (credit ρ = 0 for an all-tied prediction; `mean_spearman_tie0`,
`n_all_tied_predictions` in summary.json) is now computed for every arm. Under it the
fine-tune effects fall below significance (joint +.062 p=.32; indep +.089 p=.12).

**E3 re-checked under tie-aware scoring — robust (summary.json → ranking_paired "[tie0]"):**

| Model (all 250 videos) | joint ρ | indep ρ | Δ | Δ 95% CI | p |
|---|---|---|---|---|---|
| Qwen3.5-9B | .029 | .255 | +.226 | [.088, .363] | .0017 |
| Qwen3.5-397B | .085 | .267 | +.182 | [.046, .315] | .0135 |
| Qwen3.6-35B-A3B | .043 | .210 | +.167 | [.028, .304] | .0187 |
| Qwen3.6-27B | .163 | .329 | +.166 | [.033, .294] | .0132 |
| GPT-5 | .026 | .177 | +.151 | [.015, .286] | .0349 |
| Kimi-K2.6 | .086 | .233 | +.147 | [.019, .270] | .0299 |
| Gemini 3.7 Flash | .175 | .315 | +.139 | [.024, .254] | .0203 |

Deltas shrink (ties on INDEP now count as 0) but every CI still excludes 0 → C2 stands under
both conventions. **Paper convention: report tie-aware numbers as primary for T1 and the
pre-registered drop-ties numbers in the appendix**, stating the all-tied counts.

**Decision 2026-08-22 (owner):** GPUs unavailable ≥72 h → no run 2 before the IAEval deadline;
the paper is written on run 1 (null). Run-2 fixes (train/eval part-order alignment, 5
epochs) are pre-staged in `finetune/` and `SERVER_SESSION_PROMPT.md`; if the node frees up
before camera-ready, rerun and replace A10. Paper outline: `paper/OUTLINE.md`.

**Owner QA verdict on the rebuilt match set (2026-08-24, via the match_review site):** the
112 rebuilt tasks are hard but DISTINGUISHABLE — resolving an item takes repeated viewing
(each option compared against the reaction 1–2×), unlike the pre-rebuild set where adjacent
options were visually identical. Client-side check: min pairwise start-gap 6.0–18.1 s,
0 items <6 s (pre-rebuild median was 2.5 s). C12's quality failure is considered fixed at
the qualitative level; the quantitative gate is the G9 human confirmation pass (accuracy +
per-item confirm rate), which doubles as the T2 human reference. Revised G9 estimate:
~60–120 s/item at the owner's observed pace → 2–3.5 h over 112 items (platform resumes
across sittings). Framing note for the paper: models see one pass of sparse frames while
humans need multiple passes — reported model accuracies are under a strictly harder
input regime than the human reference.

**A10 finalized across seeds (2026-08-24; dynamo local commits 52b3c93/7bc95e3/2b97003,
independently re-computed and verified digit-for-digit):** R2 (order-aligned, 5 ep) across
seeds 0/1/2, tie-aware paired vs the local base, n=250 per seed:
JOINT Δρ = +.112 (p=.17) / +.080 (p=.23) / +.052 (p=.51) → **mean +.081 ± .030 — fails the
pre-registered ≥.10 effect bar as well**; INDEP Δρ = +.120/+.108/+.084 → +.104 ± .018
(per-seed p ≈ .05–.14). R5 ablation: the JOINT training format drives both the joint gain
and the output collapse; INDEP-only is the clean arm. Verdict: the calibration-repair
hypothesis is a **stable null across 4 training runs and 3 seeds** — the original single-seed
+0.112 was the most favorable draw. Seed2 drop-ties n=57/250 confirms the collapse artefact
(C14). Batch-1 results live as local commits on dynamo (commit-no-push discipline) — push
after owner review.

### 3.7 XV — cross-video-highlight distractor variant (added 2026-08-25)

Same correct option, distractors = other videos' trigger windows; 180 items (recovers the 68
candidates unbuildable under the pairwise-gap rule); 112 items share their correct option
with the within-video set → paired per-model comparison (`results/match/xv_vs_within.json`,
predictions `*__match_xv.jsonl`, tasks `tasks/match_xv.json`).

| Model | within (shared 112) | cross (shared 112) | Δ | McNemar p | cross (all 180) |
|---|---|---|---|---|---|
| Gemini 3.7 Flash (frames) | .500 | .536 | +.04 | .63 | .461 |
| GPT-5 | .509 | .429 | −.08 | .18 | .400 |
| Qwen3.6-35B-A3B | .482 | .366 | −.12 | .06 | .278 |
| Qwen3.6-27B | .446 | .473 | +.03 | .72 | .372 |
| Qwen3.5-397B | .411 | .375 | −.04 | .64 | .322 |
| Kimi-K2.6 | .375 | .411 | +.04 | .64 | .400 |
| Qwen3.5-9B | .259 | .277 | +.02 | .86 | .250 |

**Finding: for frame-based models, cross-video highlight distractors are NOT easier** —
deltas are small, mixed-sign, all n.s. Interpretation: the anticipated context/audio-bleed
shortcut cannot operate on silent sparse frames, and "highlight-worthy" content from other
videos is genuinely confusable. Consequences: (i) the owner's pool-expansion idea is
validated for model evaluation at current model strength (180-item cross set ≈ comparable
difficulty); (ii) within-video negatives remain the design choice for the main task because
they close the shortcut *in principle* (audio-capable models, humans) and test temporal
grounding specifically; (iii) the all-180 cross accuracies run below the shared-112 ones —
the 68 recovered items skew harder.

**Audio-shortcut probe (2026-08-26): demonstrated.** Gemini 3.7 Flash native video (audio
included) on the same XV set (`gemini-3.7-flash__video__match_xv.jsonl`, 180/180):

| arm | within (shared 112) | cross (shared 112) | Δ | McNemar b/c | p | cross (all 180) |
|---|---|---|---|---|---|---|
| native video (audio) | .545 | **.777** | **+.232** | 4/30 | **6e-6** | .761 |
| frames (no audio) | .500 | .536 | +.036 | 17/21 | .63 | .461 |

The *same model* on the *same items* gains +.232 from cross-video distractors when and only
when it can hear the audio: the reaction clip carries bleed-through of the stimulus audio, so
distractors from *other* videos can be rejected by soundtrack mismatch alone, no
reaction understanding required. Within-video negatives share the soundtrack and close this.
This settles the design question empirically: (i) the within-video contract is **necessary**,
not conservative — an XV-built set would be inflated by ≈23 points for any audio-capable
model; (ii) the frame-based protocol's modality-matched comparison is unaffected (frames Δ
n.s.); (iii) the .777 is a *shortcut* score, not a matching score, and must never be reported
as model capability.
