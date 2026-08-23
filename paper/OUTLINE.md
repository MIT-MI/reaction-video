# Paper outline — IAEval @ NeurIPS 2026 (DDL 2026-08-29/30) — written on run-1 evidence

Per-paragraph plan (co-author's method: topic + core number per paragraph, then prose).
Every number is from `gold_eval/results/summary.json` / EVIDENCE §3 + addenda; obligations
from PAPER_NOTES. Page budget: IAEval limit not stated on the tracker — assume 4 pages +
refs + appendix (NeurIPS workshop norm); trim §2 first if tighter. Status flags:
[FINAL] number is final; [PROV] provisional (T2, pending option-set rebuild); [PEND] human
study pending (judge anchor, match confirmation).

## Title (working)
ReactionBench-Gold: human-anchored evaluation of stimulus–reaction understanding
— models see reactions but cannot calibrate them

## Abstract (≤150 words, 5 sentences)
1. Problem: reaction understanding is subjective; prior benchmarks (incl. our CVPR submission)
   collapsed it into easy cross-video retrieval + LLM labels.
2. Contribution: a human-only gold set (3,257 moments × ≥3 raters, α=.357 disclosed; 200
   human rationales; 250-video ranking set) with within-video tasks and human ceilings.
3. Headline: joint within-video intensity ranking is near zero for every model (ρ ≤ .26 vs
   human .649) while independent scoring of the same moments is 2–10× better (7/7 models,
   p ≤ .035) → the failure is relative calibration, not perception.
4. Rationale: under a neutral judge no model separates (2.6–2.9/5), cue identification is the
   bottleneck; supervised LoRA on human labels does not repair calibration (null, p=.13).
5. Release: data, frozen sets, harness, ledgers.

## 1 Introduction (4 paragraphs)
P1 Why reactions: affect-from-stimulus link; what the old benchmark got wrong (cross-video
   negatives, LLM labels, top-1 on a subjective task) — one sentence each, no defensiveness.
P2 What we build: human-only gold (ADR-0001), anchored 0–3 scale (ADR-0002), three
   within-video tasks with human ceilings reported alongside every model score.
P3 What we find: C1 (joint ≈ 0), C2 (indep ≫ joint, 7/7), C4 (rationale: judge-neutral
   tie, cue bottleneck), C7 (fine-tune null + output collapse).
P4 Contributions bullet list (3): gold set + protocol; calibration finding + tie-aware
   metric; open harness with cost ledger (Gemini native video 30× cheaper, n.s. quality).

## 2 Related work (1 paragraph, ≤8 cites)
Reaction/affect datasets (MAHNOB, Reactionet, SMILE), video-LLM benchmarks, LLM-as-judge
validity + circularity; position: first within-video, human-ceiling-reported reaction benchmark.

## 3 Gold data & protocol (Methods)
P1 Source pool + pre-screen: Gemini 3.6 Flash nominations (P@≥2 .92 vs LibreFace .68, 50-clip
   blind pilot); LLM never labels (A8).
P2 Stage A: 1,999 videos / 3,257 moments, 8 raters incl. one author pass (A1), ≥3 per moment;
   α_ordinal .357; 69% of disagreement at the 1↔2 boundary (A2) → intensity is subjective,
   reported distributionally.
P3 Ground truth: consensus mean; rater-debiased model changes 0/250 orders (A3, one sentence);
   human LOO ceiling ρ=.649 on the eval set (A4); creator concentration disclosed (A5).
P4 Rationale gold: 200 moments × 3 Prolific annotators, blind write, 447 refs; unfamiliar/
   unclear as cultural signal (A7). Match set: 150 same-video 4-option items, auto-aligned
   [PROV, A6].

## 4 Tasks & evaluation protocol
P1 T1 ranking (joint / independent elicitation), T2 match (single-call MCQ; why joint
   presentation: E2 tie saturation 52–81%, B6), T3 rationale (cue/link/faithful rubric).
P2 Protocol: uniform 1 fps/512px frames; roster (B2); temp 0; CIs + paired tests (B3);
   tie-aware ρ primary (C14); three-judge T3 with leniency offset disclosed (B4).

## 5 Results
P1 Table 1 (main): T2 acc [PROV] · T1 joint ρ (tie-aware) · T1 indep ρ · T3 DeepSeek
   composite, + human ceiling row. Core numbers: Gemini-video .517 / GPT-5 .487 / … / 9B .26
   [PROV]; joint ρ .03–.18; indep .18–.33 (tie-aware); T3 2.61–2.91.
P2 Figure 1 (A3 slope chart, paired tie-aware means): every line rises; Δ +.14…+.23, all
   p ≤ .035 — "calibration, not perception"; GPT-5 joint CI includes 0.
P3 T3: no separation under DeepSeek (CIs overlap); cue ≤1.25/2 for all; judge asymmetry
   offset +1.1–1.4 (appendix table) — judge choice changes rankings, so single-judge
   leaderboards are fragile [PEND anchor validation: say "in progress"].
P4 Fine-tune (Table 2): base vs LoRA, pre-registered Δρ_JOINT +.105 p=.13 (not met); tie-aware
   +.06 p=.32; output-range collapse SD 27.9→9.8; indep gap widened .29→.39. One sentence:
   labels alone don't buy calibration.
P5 Modality & design ablations (short): native video vs frames n.s. both tasks but 30×
   cheaper (C3); per-option scoring ties (C5) justify joint MCQ; cultural strata descriptive
   only (n=22/8/120).

## 6 Limitations & ethics (1 paragraph + bullets)
Tie-drop artefact + two conventions (C14); single run per arm; T2 provisional pending rebuild
+ human confirmation; judge–human validation pending; audio confound in native arm; creator
concentration (80% four reviewers); subjectivity → report distributions, never single labels;
annotator pay; cultural unfamiliarity as signal.

## 7 Conclusion (3 sentences)
Human-anchored gold → the gap is calibration; rationale quality is judge-sensitive and
cue-limited; fine-tuning on labels is not enough → future: preference/ranking-aligned
training, match rebuild, CVPR resubmission scale.

## Appendix plan
A Table: T1 drop-ties convention + all-tied counts (C14) · B Table: E2 three conventions ·
C Table: judge overlap slices + DeepSeek column · D Table: spend/tokens per arm ·
E Prompts verbatim · F Rater effects (debias_record) · G Fine-tune training details.

## Writing rules (co-author)
common precise vocabulary (CONTEXT.md glossary); lead each paragraph with its number;
no AI-flavoured phrasing; quote only from summary.json/EVIDENCE.
