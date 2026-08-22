# Fine-tuning plan — within-video reaction intensity (decided 2026-08-21 meeting)

Owner decisions (meeting transcript 2026-08-21): fine-tune **one task only** — intensity /
within-video ranking — on the Stage-A human labels that are outside every frozen eval set
(~4/5 of the annotated pool). Match and rationale are NOT fine-tuned (match data locked on
the office machine; rationale labels too expensive). Ranking ground truth moves to
**rater-debiased consensus** (co-author's cross-annotator normalisation), framed as
preference/ranking alignment. GPUs (8×A6000) are available now; Tinker LoRA is the
fallback/parallel path. Workshop: IAEval (Evaluation of Interactive Agents), DDL 2026-08-29/30.

## 1. Why this task, and what the experiment must show

E3 (EVIDENCE §3.5) established that every zero-shot model scores moments far better in
isolation (ρ .20–.39) than jointly (ρ .03–.18), against a human ceiling of .649: the binding
failure is *within-video relative calibration*, not perception. The fine-tune asks one
question: **does supervised exposure to human-anchored intensity labels repair calibration?**
Deliverable = with/without rows for the smallest open model on T1-JOINT and T1-INDEP (same
250-video eval set, same frames protocol, same local inference path), plus zero-shot T2/T3
sanity rows to show no collateral regression. This is the rebuttal-#2 answer at workshop scale.

## 2. Data (built: dataset repo `scripts/pilot/build_finetune_split.py`)

| split | videos | moments | multi-moment videos |
|---|---|---|---|
| train | 1,364 | 1,952 | 428 |
| val (every 10th video by hashed id) | 155 | 213 | — |
| excluded (ranking 250 ∪ match ∪ rationale videos) | 480 | — | — |

Files: dataset repo `reports/phase0/finetune/{train,val}.jsonl` (+ `stats.json`). Each video row:
face_url (R2), duration, domain, creator, moments[{candidate_id, start_s, end_s, n_raters,
mean, std, median, debiased, scores}]. The platform test account (`annotator_id='test'`, 58
scores) is excluded. **Zero video overlap with any eval set.**

**Labels.** `debiased` = moment effect of the two-way additive model
score_ij = μ + moment_i + rater_j (alternating means, 30 iters, rater effects centred;
observed rater biases −0.30…+0.21 on the 0–3 scale; mean |debiased − raw mean| = 0.038).
Training target on the 0–100 scale: `round(debiased / 3 * 100)`. Raw mean kept for ablation.

**Weighting (optional, off by default):** sample weight 1/(1+std) to down-weight
high-disagreement moments; report as an ablation only if time allows.

## 3. Training task formats (both produced by `finetune/build_sft_data.py`)

Frames protocol identical to eval: 1 fps, max side 512, JPEG; windows cut from the R2 face
video with `gold_eval.frames.extract_frames(start_s, end_s)`.

**F-INDEP (primary, 1,952 train samples).** One moment per sample.
- user: [≤12 frames of the moment window] + the *exact* E3 prompt
  ("…Rate the INTENSITY … 0-100 … Reply with a single integer only.")
- assistant: `"<int>"` (e.g. `67`).

**F-JOINT (secondary, 428 train samples — the calibration objective).** One video per sample,
only videos with ≥2 moments.
- user: [≤32 frames of the full face video, evenly subsampled] + the *exact* T1 joint prompt
  with the moments list (ids + time ranges).
- assistant: `{"m0": 67, "m1": 21, ...}` — integers from debiased labels.
  (Eval uses ≤64 frames; training at ≤32 keeps sequence length ≤~8k tokens; the eval-time
  64-frame input is a mild length extrapolation — report it, and optionally eval at 32 too.)

Mixture: INDEP + JOINT concatenated, JOINT oversampled ×2 so it is ~30% of steps.
Ablation rows if time: INDEP-only, JOINT-only.

**Prompt/answer strings are imported from `gold_eval/run_ranking_independent.py` and
`gold_eval/run_ranking.py` — never retyped — so train and eval prompts cannot drift.**

## 4. Model & optimisation (`finetune/train_lora.py`)

- Base: `Qwen/Qwen3.5-9B` (the roster's smallest open VLM; same family as the Tinker arms).
  Secondary if time: `Qwen/Qwen3.6-35B-A3B` (MoE, fits 8×48GB with LoRA).
- LoRA (PEFT): r=16, α=32, dropout 0.05 on the **language model's** q/k/v/o/gate/up/down
  projections; vision tower + merger frozen. Trainable ≈ 40–60M params.
- bf16, gradient checkpointing, per-device batch 1, grad-accum 4 → effective 32 across 8
  GPUs (torchrun DDP). LR 1e-4 cosine, warmup 3%, 3 epochs over the mixture (~7.5k samples
  → ~700 optimizer steps). Loss on assistant tokens only.
- Early stopping on val loss (val = 155 videos, both formats); keep best checkpoint.
- Seeds: 0 (+1, 2 if budget allows → report mean±std; otherwise single run, flagged C9).
- Wall-clock estimate: ≤12 frames/sample ≈ 2–3k tokens, 32 frames ≈ 6–8k tokens; 9B LoRA
  on A6000 ≈ 1–2 s/sample → ~3 h for 3 epochs. Plan for one evening.

## 5. Evaluation protocol (the comparison that goes in the paper)

Inference path: `gold_eval/backbones.py::HFLocalBackbone` (transformers, greedy) for BOTH
base and LoRA — the base model must be re-evaluated through the same local path so the
with/without contrast has no Tinker-vs-local confound (the Tinker base numbers stay in the
main table as the zero-shot row; the local base row is the FT control).

Runs on the frozen 250-video ranking set, frames protocol unchanged:
1. `run_ranking --model hf:<base>` and `hf:<base>+<lora>` → T1-JOINT ρ (+ CI, paired Wilcoxon
   base vs FT via `summarize.py ranking_paired`-style code).
2. `run_ranking_independent` for both → T1-INDEP ρ.
3. Zero-shot collateral: `run_match` (T2, after the option-set rebuild) and `run_rationale`
   + judge for the FT model — expect ≈unchanged; regression would be a finding too.
4. Report against the human LOO ceiling (.649) and the Tinker zero-shot rows.

Success criterion (pre-registered here): FT improves T1-JOINT mean ρ over the local base by
≥0.10 with Wilcoxon p<.05 on paired videos. A null result is reported as such.

## 6. Ground-truth normalisation follow-up (co-author request)

The eval set's consensus is currently the raw mean (+ median for pass/fail). Add
`debiased` to the ranking set *without changing membership* (compute for the frozen 250
videos and report ρ against both raw-mean and debiased consensus; expected near-identical
given |Δ|≈0.04). Do NOT rebuild the ranking set: removing the test account changes
spreads/selection, and all T1 numbers are on the frozen build (EVIDENCE C13).

## 7. Schedule (DDL 2026-08-29/30)

| when | step |
|---|---|
| now | data split built ✔; `build_sft_data.py` smoke-tested locally on a few videos |
| GPU day 1 (today/tomorrow) | clone gold-eval branch on the node, venv, `build_sft_data.py` full (downloads ~1.5k face videos from R2, ~1–2 h), launch `train_lora.py` (~3 h) |
| GPU day 1 evening | eval base + LoRA via HFLocalBackbone on T1-JOINT/INDEP (250 videos ×2 ×2 ≈ 1 h) |
| day 2 | zero-shot T3 + (post-rebuild) T2 rows; ablations if time; EVIDENCE A10 row |
| Mon 8/24 | office machine → match rebuild (separate track) |

## 8. Risks

- HF weights/processor API for Qwen3.5-9B: `train_lora.py` uses `AutoProcessor` +
  `AutoModelForImageTextToText`; if the class differs, adjust the two loader lines.
- Sequence length: cap frames per sample (12 / 32) and use gradient checkpointing; drop
  samples >10k tokens rather than truncating images.
- Label noise: α≈.36 on the 0–3 scale; debiasing + ≥3-rater averaging mitigates; the
  1/(1+std) weighting is the lever if val loss plateaus early.
- Tinker fallback: identical SFT JSONL feeds `tinker_cookbook` supervised recipe with
  ImagePart inputs; use if the node is pre-empted.
