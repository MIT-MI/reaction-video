# Claim–evidence architecture — IAEval @ NeurIPS 2026 (built 2026-08-22)

Derived from `gold_eval/EVIDENCE_2026-08.md` (+ addenda 08-21/08-22), `PAPER_NOTES.md`,
`results/summary.json`, `finetune/REPORT.md`, storyline `paper/OUTLINE.md`. Old CVPR numbers
ignored. Policy context: `ReactionBench-paper/paper_context.yaml` (integrity-core +
academic-defaults; double-blind; draft; page pressure high; evidence maturity partial).
Strength: S = direct measurement with CI/paired test · M = "indicates / consistent with" ·
W = scoped/descriptive only.

## 1. Candidate claims

Artifact
- K1 (S) Human-only gold: 3,257 moments / 1,999 videos, ≥3 anchored 0–3 ratings each,
  α_ordinal=.357 disclosed; 200 moments × 3 human rationales (447 refs); 250-video ranking set;
  150-item same-video match set. LLMs never produced a gold label.
- K2 (S) Ranking GT robust to rater bias: two-way debiasing changes order in 0/250 videos; all ρ
  identical to 4 dp.
- K3 (S) Human LOO ceiling on the ranking eval set ρ=.649 (420 rater–video pairs); pool-specific.
- K4 (M) Disagreement is structured: 69% of splits at the 1↔2 boundary → report distributionally.

Substantive
- K5 (S) Joint within-video ranking near zero for every model: tie-aware ρ .03–.18 (Gemini
  native video .26); GPT-5 CI [−.08,.14] includes 0; human .649.
- K6 (S) Independent scoring beats joint for all 7 models on the same videos: tie-aware Δρ
  +.14…+.23, all CIs exclude 0, p≤.035; drop-ties convention agrees.
- K7 (M) Under one neutral judge (DeepSeek-V3.1) rationale quality does not separate models:
  composites 2.61–2.91/5, CIs overlap.
- K8 (M) Cue identification is low for every model (≤1.25/2, neutral judge).
- K9 (M) Judge choice shifts absolute scores +1.1–1.4 on identical generations (Gemini vs GPT-5
  judge; rank agreement r .76–.85).
- K10 (M) LoRA on 1,952 human-labelled moments did not detectably repair joint calibration:
  Δρ_JOINT +.105, p=.13 (criterion not met); tie-aware +.06, p=.32; output SD 27.9→9.8,
  84/250 all-tied.
- K11 (W/M) Match accuracy 26–52%, orders roughly with scale — PROVISIONAL (C12).

Design-justification / control
- K12 (S) Per-option scoring of same-video candidates ties on 52–81% (SBS 61%) → same-video
  negatives need joint presentation.
- K13 (S, narrow) Native video (Gemini only) ≈30× fewer input tokens, no significant accuracy
  loss, non-significant positive direction on both tasks.
- K14 (S) Scorer artefact: dropping all-tied predictions inflates mean ρ for collapsing arms;
  tie-aware convention removes it and K6 survives.
- K15 (W) Cultural strata: no visible gap (n=22/8/120), descriptive only.

## 2. Necessary evidence per claim

| Claim | Evidence (present unless marked) | Source |
|---|---|---|
| K1 | Stage A stats; Stage C completion; frozen-set builders + selection records | dataset stats.json, selection_record.json |
| K2 | debias_record.json; summary.json→mean_spearman_debiased | EVIDENCE add. 08-21 |
| K3 | LOO ceiling on 250-video set | §3.2 |
| K4 | Stage A disagreement breakdown | GOLD_STAGE_A_ANALYSIS.md |
| K5 | T1 joint, 7 arms + native; boot CIs; tie-aware means | §3.2, §3.5, add. 08-22 |
| K6 | T1 indep; paired Wilcoxon + boot CI, both conventions | §3.5 G1 + tie0 table |
| K7,K8 | T3 gens + DeepSeek judge column with CIs | G5 |
| K9 | 60-gen dual-judged overlap slices | §3.4 |
| K10 | finetune_paired.json, REPORT.md, local-path base control | add. 08-22 |
| K11 | T2 MCQ + Wilson CIs — NEEDS rebuilt option sets + re-run + G9 confirmation | §3.1, C12, G9 |
| K12 | E2 per-option + V3 SBS, three conventions | §3.3 |
| K13 | E1 paired McNemar/Wilcoxon + token ledger | §3.5, costs |
| K14 | n_all_tied_predictions per arm; E3 under both conventions | add. 08-22 |
| K15 | T2 strata | §3.1 |

## 3. Experiments that only support weaker wording

| Experiment | Cannot support | Defensible wording |
|---|---|---|
| T2 match | absolute accuracy / ordering (98/150 near-duplicate distractors; auto GT) | "provisional" or omit from central claims until rebuild |
| E1 native video | "video beats frames"; beyond Gemini; visual vs audio (C5) | "n.s. at n=150/249; ~30× fewer input tokens; direction positive; one model" |
| T3 (any judge) | judge validity (G3 pending); fine ordering; "cue is THE bottleneck" (GPT-5 cue 1.25 > link 1.07) | "no separation under a neutral judge; cue ≤1.25/2; judge–human validation in progress" |
| T3 mixed-judge columns | cross-column leaderboard | within-judge or DeepSeek column only |
| Fine-tune run 1 | "fine-tuning cannot fix calibration"; effect-size claims (point est. cleared bar; single seed; 24-vs-64 frames; part-order asymmetry) | "did not detectably improve calibration at this scale and seed; output range collapsed" |
| E2 "below chance" | models below chance | tie saturation → joint presentation needed; all three conventions |
| Cultural strata | significance | one descriptive sentence |
| Raw INDEP per-file means | cross-model ranking (coverage 182–241 differs) | tie-aware all-250 / paired numbers; gaps <.05 = noise |
| "2–10× better" ratio | ratios on near-zero denominators | Δρ with CI |

## 4. Recommended central claims

1. **Calibration, not detection, is the binding failure on reaction intensity.** K5+K6+K14+K3.
2. **Human-only, within-video gold with disclosed reliability and ceilings makes this
   measurable** (artifact). K1–K4, K12.
3. **Supervised intensity labels did not detectably buy calibration** (null, bounded). K10.
   Keep only with single-seed/part-order caveats in the same paragraph; else demote to a
   results paragraph and let T3 judge-sensitivity (K7+K9) be third.
Spine = 1+2; T3 stays supporting until G3 is scored.

## 5. Claim–experiment matrix

● necessary & present · ◐ present but provisional/limited · ○ supportive · ✗ missing

| | T1 joint | T1 indep+paired | tie-aware | LOO ceiling | debias | Stage A stats | T3+DeepSeek | judge overlap | G3 anchor | T2 MCQ | T2 rebuild+G9 | E2/V3 | E1 paired+ledger | FT run1 | FT run2/seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K1 | | | | ○ | | ● | | | | ◐ | ✗ | | | | |
| K2 | ○ | | | | ● | | | | | | | | | | |
| K3 | | | | ● | | | | | | | | | | | |
| K4 | | | | | | ● | | | | | | | | | |
| K5 | ● | | ● | ● | ○ | | | | | | | | | | |
| K6 | ● | ● | ● | ○ | | | | | | | | | | | |
| K7 | | | | | | | ● | ○ | ✗ | | | | | | |
| K8 | | | | | | | ● | | ✗ | | | | | | |
| K9 | | | | | | | ○ | ● | ○ | | | | | | |
| K10 | ● | ● | ● | ○ | | | | | | | | | | ● | ✗ |
| K11 | | | | | | | | | | ◐ | ✗ | | | | |
| K12 | | | | | | | | | | ○ | | ● | | | |
| K13 | ○ | | | | | | | | | ◐ | | | ● | | |
| K14 | ● | ● | ● | | | | | | | | | | | ● | |
| K15 | | | | | | | | | | ◐ | ✗ | | | | |

## 6. Unresolved claim–evidence mismatches (owner decisions needed)

1. Title/abstract "models *see* reactions but cannot calibrate" — perception is not measured;
   INDEP ρ .18–.33 ≤ half the ceiling. Write "independent scoring recovers part of the signal
   joint ranking loses".
2. Ceiling vs elicitation mode: .649 comes from Stage-A per-moment ratings; A3 caption says
   "measured on the joint task". Confirm from the annotation UI whether raters saw full-video
   context; state it, or comparing joint ρ to the ceiling is apples-to-oranges.
3. "2–10× better" in outline abstract → Δρ with CIs.
4. "Cue is the bottleneck" not uniform under DeepSeek → "cue scores low for all models".
5. T3 claims without G3 anchoring are claims about judge-scored quality only; carry [PEND]
   into prose.
6. T2 in the main table with 98/150 flawed option sets and no human confirmation invites last
   year's well-posedness critique → rebuild before 8/29 or move T2 to appendix with disclosure.
7. Fine-tune "labels don't buy calibration" is a single-seed null with point estimate above
   the bar and two train/eval asymmetries → "did not detectably", caveats adjacent; run 2 if
   GPUs free before camera-ready.
8. Novelty "first within-video, human-ceiling-reported reaction benchmark" — no related-work
   comparison / .bib yet; unsupported until citation audit.
9. K13 "30× cheaper" = input tokens, one model, audio included → scope; drop from
   contributions list (outline P4).
10. Creator concentration (4 reviewers = 80%) with no per-creator slice (G6) → compute a
    descriptive slice or name the pool in K5/K6 scope.
11. Prescreen self-selection (G6 not run): Gemini 3.6 nominated moments, Gemini 3.7 evaluated
    → at least a stated limitation for Gemini-involving comparisons.

Next: paper-figures-tables for A1 (T2 column per item 6) and A3 slope figure from tie-aware
paired means; paper-writing prose only after items 1, 2, 6 are decided.
