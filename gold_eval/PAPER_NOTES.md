# PAPER_NOTES — the single registry of things the paper MUST state

Canonical list of methodological facts, disclosures, robustness checks and caveats that
must appear in the manuscript. Each item: what to write, where it goes, where the evidence
lives. Keep this file current; EVIDENCE_2026-08.md holds the numbers, this file holds the
obligations. (Owner request 2026-08-21: "找到一个统一的地方把这些重要事项都记录下来".)

## A. Annotation & ground truth (Methods / Data section)

| # | Must state | Evidence |
|---|---|---|
| A1 | Stage A: 3,257 moments / 1,999 videos, every moment ≥3 raters (most 4–5), 8 raters incl. **one author pass** (`annotator_id='test'`, 58 scores; ADR-0002 permits author raters, blind to model outputs) — disclose explicitly. | dataset `docs/gold_stage_a_figures/stats.json`; EVIDENCE C13 |
| A2 | Reliability = Krippendorff α_ordinal **0.357**; "within-1 agreement" retired because chance within-1 is 91% on this skewed 4-point scale. Disagreement is structured: 69% of splits at the 1↔2 boundary, 7% at 0↔1. | `GOLD_STAGE_A_ANALYSIS.md` |
| A3 | **Ranking ground truth = rater-debiased consensus** (two-way additive model score = μ + moment + rater; rater effects −0.30…+0.21). Robustness check: debiasing changes within-video order in **0/250** eval videos and leaves every model ρ identical to 4 decimals → ranking GT is robust to cross-annotator bias (co-author's normalisation concern, resolved). Report the raw-mean numbers, cite the check in one line. | dataset `reports/phase0/ranking/debias_record.json`; `summary.json → mean_spearman_debiased` |
| A4 | Human ceilings are pool-specific: full pool r≈.51 / ρ≈.40 vs curated 250-video eval set LOO ρ=**.649** (420 rater-video pairs). Always name the pool. | EVIDENCE C11 |
| A5 | Ranking eval set: 250 videos, rankable = ≥2 moments with consensus spread ≥0.25; creator cap relaxed 35→50 → **4 Patreon reviewers = 80%**; disclose creator concentration (~30 creators overall, top 4 = 72% of pool). | `selection_record.json` |
| A6 | Match ground truth = temporal alignment (auto), match-ready filter; **option sets being rebuilt** (98/150 tasks had two options starting <4 s apart); all T2 numbers are provisional until rebuild + re-run + human 1-click confirmation. If rebuilt set <150, state the count. | EVIDENCE C12/G9 |
| A7 | Rationale gold: 200 moments, 3 Prolific annotators completed 200/200 each (+2 abandoned after 4 items); 447 'ok' references; **unclear/unfamiliar are recorded as signals** (one annotator opted out of 51%, cross-validated: of their 54 'unclear', 31 were written by both others) — this is the cultural-situatedness design (§7 of STATUS plan), not a defect. Annotators paid regardless. | Supabase `gold_rationales`; EVIDENCE §1 |
| A8 | Pre-screening by Gemini 3.6 Flash (Precision@≥2 ≈ .92 vs LibreFace .68 in a 50-clip blind pilot, pre-registered adoption rule) — LLM never produced a gold label (ADR-0001). Evaluated Gemini (3.7 Flash) is the successor; disclose and report the prescreen-rank self-selection check (G6, not yet run). | `PHASE0_PROGRESS_2026-07.md`; ADR-0001/0003 |

## B. Evaluation protocol (Experiments section)

| # | Must state | Evidence |
|---|---|---|
| B1 | Uniform input budget: 1 fps, max side 512, caps (match 12/6, ranking 64, rationale 12, indep 12); temperature 0; GPT-5 reasoning_effort=minimal; Gemini thinking_level=low. | EVIDENCE §2 |
| B2 | Roster: GPT-5, Gemini 3.7 Flash (frames + native video), Qwen3.5-9B / 3.6-27B / 3.6-35B-A3B / 3.5-397B-A17B, Kimi-K2.6 (open weights via Tinker); Claude, Gemini Pro, GPT-4o dropped (stale/owner decision) — say so in a footnote. | ADR-0003 |
| B3 | Single run per arm; report Wilson CIs for accuracies and seeded bootstrap CIs for mean ρ; paired tests (Wilcoxon / McNemar) for within-model comparisons. **T1 ρ is reported tie-aware (all-tied prediction = ρ 0) as primary; drop-ties convention in appendix with all-tied counts** (EVIDENCE C14). | `summary.json` (`mean_spearman_tie0`) |
| B4 | **T3 judging is three-judge**: GPT-5 (judges non-OpenAI), Gemini (judges GPT-5), and a neutral-family **DeepSeek-V3.1** column scoring all 7 — the only column that is directly comparable across models. Disclose the leniency offset (Gemini judge ≈ +1.1–1.4 vs GPT-5 judge on identical generations; rank agreement r .76–.85). | EVIDENCE §3.4 + addendum |
| B5 | Judge–human validation: 84-item anchor set (7 models × 3 judge-score strata × 4), human rubric identical to the judge's; report correlation once scored (G3). Until then say "validation in progress / limitation". | `gold_anchor_scores` |
| B6 | Match per-option scoring (old protocol) must be reported with three conventions (strict tie=wrong / random tie-break expectation / decided-only), never strict alone. | EVIDENCE §3.3 |

## C. Findings that are safe to claim (and their scope)

| # | Claim (narrowest defensible form) | Evidence |
|---|---|---|
| C1 | Within-video joint ranking is near-zero for every model (ρ ≤ .26; GPT-5 CI includes 0) vs human .649. | §3.2 |
| C2 | Independent per-moment scoring beats joint for all 7 models — tie-aware Δρ +.14…+.23 (p ≤ .035), drop-ties Δρ +.15…+.32 (p ≤ .03), CIs exclude 0 under both: the binding failure is within-context relative calibration, not perception. | §3.5 G1 + 2026-08-22 addendum |
| C3 | Native video vs frames (Gemini): directionally better on both tasks but **n.s.** (McNemar p=.36, Wilcoxon p=.27); ~30× fewer input tokens (deterministic). Do NOT claim "video beats frames". | §3.5 |
| C4 | Under a single neutral judge, rationale quality does not separate models (composites 2.61–2.91, overlapping CIs); cue identification is the hard dimension (<1.25/2 for all). | §3.4 DeepSeek column |
| C5 | Same-video hard negatives are only discriminable under joint presentation (per-option scoring ties on 52–81% of items) — design justification, not a below-chance claim. | §3.3 |
| C6 | Match accuracy orders with capability (26%→52%) and is unsaturated — **provisional until option-set rebuild**. | §3.1 + C12 |
| C7 | **Fine-tune = stable null across 3 seeds + 2 protocols**: order-aligned 5-ep LoRA gives JOINT Δρ mean +.081±.030 (tie-aware; no seed p<.05) — below the pre-registered ≥.10 bar; INDEP +.104±.018. JOINT-format training drives both the small gain and the output collapse (R5). Write as a finding: human intensity labels alone do not buy within-video calibration at this scale. | dynamo finetune/REPORT_R2.md §R6 (commit 2b97003); REPORT_R5.md |

## D. Limitations to write (not hide)

Tie-drop scorer artefact and the two T1 conventions (C14); fine-tune single seed and 24-vs-64 frame train/eval length gap; creator concentration (A5); cultural strata tiny (n=22/8/120, descriptive only); single run
per arm; T2 GT auto-aligned pending human confirmation; judge–human validation pending;
native-video arm includes audio (modality confound); 10 empty Gemini rationale generations
(n=191); E3 coverage gaps handled by paired-intersection analysis.

## E. Writing guidance (co-author, meeting 2026-08-21)

- Use precise, *common* terminology — avoid AI-sounding or idiosyncratic terms; unify vocabulary
  first (CONTEXT.md glossary), then write.
- Be concise; lead with the core finding of each paragraph; results/experiments sections
  start from a per-paragraph outline (topic + core number) before prose.
- Every number quoted from `summary.json` / EVIDENCE §3 — never recomputed ad hoc.
