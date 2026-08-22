# LoRA fine-tune run 1 — within-video reaction intensity (Qwen3.5-9B)

Executed 2026-08-22 against `finetune/PLAN.md`. Every number below comes from a file under
`gold_eval/results/` or `finetune/ckpt/Qwen_Qwen3.5-9B-lora-r16/train_args.json`; nothing is
back-filled or estimated.

---

## Setup

| | |
|---|---|
| Node | `ise-dynamo.cs.illinois.edu`, Linux 5.15.0-161-generic, 96 vCPU, 503 GB RAM |
| GPUs | 8 × NVIDIA RTX A6000, 49,140 MiB each (~47.4 GiB usable), driver 575.57.08 (CUDA 12.9) |
| Python | 3.13.13 (venv at `reaction-video/.venv`) |
| torch | 2.11.0+cu128 (torchvision 0.26.0+cu128) |
| transformers | 5.15.1 |
| peft | 0.20.0 |
| accelerate | 1.14.0 |
| numpy / scipy / pillow | 2.5.2 / 1.18.1 / 12.3.0 |
| ffmpeg | 4.4.2-0ubuntu0.22.04.1 |
| Base model | `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

**Loader edits: none were needed.** PLAN §8 anticipated that `AutoProcessor` /
`AutoModelForImageTextToText` might need class-name adjustment. On transformers 5.15.1 both
resolve correctly for `model_type: qwen3_5` — `AutoProcessor → Qwen3VLProcessor`,
`AutoModelForImageTextToText → Qwen3_5ForConditionalGeneration`. The two loader lines are
unchanged. Other edits were required to make the (previously unrun) path execute at all; they
are enumerated under **Deviations**.

A first `pip install "torch"` produced a cu130 build that the 12.9 driver rejects
(`torch.cuda.is_available() == False`); reinstalled from the cu128 index. `torchvision` is a
hard requirement of `Qwen3VLVideoProcessor` and had to be added.

---

## Data

`python -m finetune.build_sft_data --split .../reports/phase0/finetune --joint_cap 24`

| split | INDEP rows | JOINT rows (oversampled ×2 on train) | total | failed videos |
|---|---|---|---|---|
| train | 1,952 | 856 (from 428 multi-moment videos) | 2,808 | **0** |
| val | 213 | 48 (×1) | 261 | **0** |

Matches PLAN §2/§3 exactly (1,364 train / 155 val videos; 1,952 / 213 moments). All 3,069 rows
validated: every image path exists on disk, every INDEP answer is an integer in 0–100, every
JOINT answer parses as JSON with integer values in 0–100.

**Token lengths** (120 sampled rows per split×format, tokenised through the real `SFTData`):

| split | format | min | p50 | p90 | max | mean |
|---|---|---|---|---|---|---|
| train | indep | 614 | 872 | 1,130 | 1,130 | 947 |
| train | joint | 4,601 | 6,410 | 6,435 | 6,485 | 6,400 |
| val | indep | 613 | 872 | 1,130 | 1,130 | 947 |
| val | joint | 5,118 | 6,410 | 6,435 | 6,458 | 6,388 |

**No sample exceeded the 10,000-token skip threshold**, so PLAN §8's drop path never fired and
no training row was silently discarded.

Frames per sample: JOINT median 24 (capped, see Deviations); INDEP min 2 / median 3 / max 4 —
moment windows are ~3 s at 1 fps, so **the 12-frame INDEP cap never binds** and `--indep_cap`
is inert at any value ≥4.

---

## Training

`torchrun --nproc_per_node 8 -m finetune.train_lora --base Qwen/Qwen3.5-9B --epochs 3 --lr 1e-4 --rank 16`

| | |
|---|---|
| Trainable params | **29,097,984** (0.3083% of 9,438,911,728) |
| Optimizer steps | 264 (3 epochs; effective batch 32 = 8 GPUs × 1 × grad-accum 4) |
| LR schedule | cosine, warmup 3%, peak 1e-4 |
| Wall-clock | **11,430 s ≈ 3.18 h** |
| Train loss | 0.967 (first logged) → 0.365 (last logged); run mean 0.4635 |
| Val loss | step 100 → 0.46458; step 200 → 0.40885; **step 264 → 0.39549 (best)** |
| Best checkpoint | step 264 (the final step) |

Trainable params came in **below PLAN §4's 40–60M estimate**, and the reason is structural, not
a misconfiguration: Qwen3.5-9B is a hybrid-attention model where 24 of 32 layers are
`linear_attn` exposing `in_proj_qkv` / `in_proj_z` / `in_proj_b` / `in_proj_a` / `out_proj`.
PLAN §4's target set is "q/k/v/o/gate/up/down projections", and the regex implementing it
matches only the 8 full-attention layers' q/k/v/o plus all 32 MLPs — 128 modules. The 40–60M
figure assumed dense attention in every layer. I did **not** widen the target set, since that
would change a pre-registered design decision. Verified: zero vision-tower modules are adapted
(all 256 trainable tensors are under `language_model`), so PLAN §4's "vision tower + merger
frozen" holds.

Loss masking was verified to land on the assistant span only. With the fix described under
Deviations, the unmasked span is exactly the answer plus the turn terminator:

```
indep row  seq_len=872   unmasked=4   decoded='33<|im_end|>\n'      (answer '33'  = 2 tokens)
indep row  seq_len=1130  unmasked=4   decoded='54<|im_end|>\n'
joint row  seq_len=2794  unmasked=18  decoded='{"m0": 60, "m1": 60}<|im_end|>\n'
```

Val loss was still falling at the final step, so `EarlyStoppingCallback` never triggered and
`load_best_model_at_end` selected the last checkpoint. With only 3 evaluations (eval_steps=100
over 264 steps) early stopping could not have fired regardless. **3 epochs looks like
under-training rather than over-training** on this mixture.

---

## Results

Frozen 250-video ranking set, frames protocol untouched (JOINT ≤64 frames, INDEP window ≤12),
both arms through `HFLocalBackbone` with greedy decoding. Consensus = Stage-A raw mean; ρ
against the rater-**debiased** consensus is identical to 4 dp for all four runs
(PLAN §6 asked for both — `mean_spearman_debiased` in `summary.json` equals `mean_spearman`).

### Per-arm (source: `gold_eval/results/summary.json`)

| arm | T1-JOINT mean ρ [95% CI] | n scored | T1-INDEP mean ρ [95% CI] | n scored |
|---|---|---|---|---|
| base, local path | 0.0236 [−0.0862, 0.1344] | 250 | 0.3184 [0.1998, 0.4334] | 199 |
| **LoRA r16** | **0.1291** [−0.0077, 0.2620] | **166** | **0.5217** [0.4123, 0.6281] | **164** |
| *Tinker zero-shot Qwen3.5-9B (reference)* | *0.0294* [−0.0799, 0.1366] | *250* | *0.3340* [0.2160, 0.4482] | *191* |
| *Human LOO ceiling (reference)* | *0.649* (420 rater-video pairs) | — | *0.649* | — |

Unparsed responses: 0 in all four runs. The local base row lands on top of the Tinker zero-shot
row for the same model (.0236 vs .0294 joint; .3184 vs .3340 indep), confirming PLAN §5's
requirement that the with/without contrast carries no Tinker-vs-local confound. Human ceiling
from `gold_eval/EVIDENCE_2026-08.md` §3.2 / `PAPER_NOTES.md` A4; Tinker rows from
`summary.json`.

**The n-scored columns are not comparable across rows** — see the paired test, which is the
statistic that controls for this.

### Pre-registered paired test (source: `gold_eval/results/finetune_paired.json`)

Per-video Spearman from `score_ranking.per_video_rhos`, intersected over videos both arms
scored; Wilcoxon signed-rank + seeded percentile bootstrap CI of the mean Δ (`RNG_SEED=0`,
`N_BOOT=10000`, the same estimator `summarize.ranking_paired` uses).

| task | n paired | base ρ | LoRA ρ | **Δρ** | Δ 95% CI | Wilcoxon p | improved / worsened / tied |
|---|---|---|---|---|---|---|---|
| **T1-JOINT** | 166 | 0.0239 | 0.1291 | **+0.1052** | [−0.0216, 0.2309] | **0.131** | 42 / 29 / 95 |
| T1-INDEP | 136 | 0.3418 | 0.5262 | **+0.1844** | [0.0840, 0.2927] | **0.0031** | 36 / 17 / 83 |

> **Pre-registered criterion — Δρ_JOINT ≥ 0.10 with p < 0.05: NOT MET.**
> The effect-size bar is cleared (+0.105) but the significance bar is not (p = 0.131).

### Output-range collapse and the scoring artefact it creates

| task | arm | all-tied (unrankable) predictions | prediction SD | distinct values emitted | n scored |
|---|---|---|---|---|---|
| T1-JOINT | base | 0 / 250 | 27.89 | 28 | 250 |
| T1-JOINT | LoRA | **84 / 250** | 9.81 | 12 | 166 |
| T1-INDEP | base | 51 / 250 | 22.97 | 16 | 199 |
| T1-INDEP | LoRA | **86 / 250** | 10.80 | 10 | 164 |

The fine-tuned model compressed its output distribution toward the training-label mean. When a
model emits identical scores for every moment in a video, Spearman is undefined and the
pre-registered scorer **drops that video** — so collapse removes the LoRA's least informative
cases from its own average rather than penalising them. That is why LoRA's n scored falls from
250 to 166 on JOINT while the base loses none.

**Post-hoc sensitivity check** (not pre-registered, reported because the two arms tie at very
different rates): credit ρ = 0 — chance — to an all-tied prediction, over all 250 videos with
non-tied gold.

| task | n | base ρ | LoRA ρ | Δρ | Δ 95% CI | Wilcoxon p |
|---|---|---|---|---|---|---|
| T1-JOINT | 250 | 0.0236 | 0.0857 | +0.0621 | [−0.0422, 0.1655] | 0.320 |
| T1-INDEP | 250 | 0.2534 | 0.3423 | +0.0888 | [−0.0020, 0.1789] | 0.117 |

Under this accounting **neither task's improvement is significant**, and the INDEP effect that
passes the pre-registered test does not survive.

---

## Honest interpretation

The pre-registered criterion was **not met**: T1-JOINT moved by Δρ = +0.105, clearing the
≥0.10 effect-size bar, but at Wilcoxon p = 0.131 over 166 paired videos the joint gain is not
distinguishable from zero. T1-INDEP did improve significantly on the pre-registered paired test
(Δρ = +0.184, p = 0.003), but that is the axis that was already strong, and the
joint-versus-independent gap **widened** rather than closed (0.294 → 0.393), so supervised
exposure to human-anchored labels did not repair the within-video relative calibration that E3
identified as the binding failure. The dominant behavioural change is output-range collapse
toward the training-label mean — prediction SD fell 27.9 → 9.8 (joint) and 23.0 → 10.8
(indep), with the LoRA emitting unrankable all-tied scores on 84/250 joint videos against 0 for
the base — and because the pre-registered scorer discards undefined-Spearman videos, that
collapse flatters the LoRA's headline average by silently shrinking its denominator. A
tie-aware post-hoc check that credits ρ = 0 for uninformative predictions shrinks both effects
below significance (joint +0.062, p = 0.32; indep +0.089, p = 0.12), which I read as the more
faithful summary. This is therefore a **null result for the calibration hypothesis at this
scale**, from a single seed (C9), and the eye-catching .32 → .52 independent-ranking number
must not be quoted without its n and its tie caveat.

---

## Exact commands run

```bash
# clones (side by side)
git clone -b gold-eval https://github.com/MIT-MI/reaction-video.git
git clone -b phase0/pilot https://github.com/quao627/reaction_video_benchmark.git \
    reaction_video_benchmark_phase0_pilot
cd reaction-video

# env
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install "torch" "transformers>=4.50" "peft" "accelerate" "pillow" "scipy" "numpy"
.venv/bin/pip install --force-reinstall "torch" --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install torchvision --index-url https://download.pytorch.org/whl/cu128

# 1a. SFT data — smoke, then full
.venv/bin/python -m finetune.build_sft_data \
    --split ../reaction_video_benchmark_phase0_pilot/reports/phase0/finetune --limit 3
.venv/bin/python -m finetune.build_sft_data \
    --split ../reaction_video_benchmark_phase0_pilot/reports/phase0/finetune            # cap 32, OOM'd
.venv/bin/python -m finetune.build_sft_data \
    --split ../reaction_video_benchmark_phase0_pilot/reports/phase0/finetune --joint_cap 24

# 1b. training (8 × A6000)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/torchrun --nproc_per_node 8 -m finetune.train_lora \
    --base Qwen/Qwen3.5-9B --train finetune/data/train.jsonl --val finetune/data/val.jsonl \
    --out finetune/ckpt/Qwen_Qwen3.5-9B-lora-r16 --epochs 3 --lr 1e-4 --rank 16

# 1c. eval — both arms through the same local path (base run BEFORE training, GPUs were idle)
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m gold_eval.run_ranking             --model "hf:Qwen/Qwen3.5-9B"
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python -m gold_eval.run_ranking_independent --model "hf:Qwen/Qwen3.5-9B"
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m gold_eval.run_ranking \
    --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/Qwen_Qwen3.5-9B-lora-r16/best"
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python -m gold_eval.run_ranking_independent \
    --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/Qwen_Qwen3.5-9B-lora-r16/best"
.venv/bin/python -m gold_eval.summarize

# 1d. paired test
.venv/bin/python -m finetune.paired_test --lora finetune/ckpt/Qwen_Qwen3.5-9B-lora-r16/best
```

`finetune/run_node.sh` was not used as a single shot; its stages were run by hand so each could
be inspected, and the base eval was moved ahead of training to use otherwise-idle GPUs.

---

## Deviations from PLAN.md

**Nothing under `gold_eval/tasks/` was touched. No evaluation prompt string and no frame
protocol was altered.** The E3 and T1 prompts are still imported from the eval runners by
`build_sft_data.py`, so train and eval strings cannot drift.

### Substantive — affects how the result should be read

1. **JOINT frame cap 32 → 24** (PLAN §8's named remedy). At 32 frames training OOM'd at step 6:
   `tried to allocate 7.89 GiB, 2.14 GiB free`. The dominant allocation is the fp32 logits
   tensor, vocab 248,320 × ~8.5k tokens ≈ 8.5 GB, on top of 19 GB of weights. Consequence:
   training sees ≤24 frames while eval still sees ≤64, **widening** the length extrapolation
   PLAN §3 already flagged. This works against the fine-tune, so it is a caveat on the null
   result rather than an explanation of the positive INDEP number (INDEP frame counts are
   unaffected — see below).
2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.** The failing allocation had 8.41 GiB
   "reserved but unallocated", i.e. fragmentation rather than a genuine capacity wall. This
   alone carried a worst-case 32-frame batch through the training step, but
   `load_best_model_at_end` still OOM'd on two ranks, which is why (1) was also needed. Before
   relaunching I validated the fix on a purpose-built worst case — 32 × 24-frame joint samples,
   one full grad-accum step of nothing but the longest rows — which completed and saved cleanly.
3. **Thinking disabled at generation time in `HFLocalBackbone`** (`enable_thinking=False`).
   Qwen3.5's chat template opens a `<think>` block, so with `run_ranking_independent`'s
   `max_tokens=8` budget the model spent every token reasoning and emitted no digit — **zero
   scores would have parsed for either arm**. `TinkerBackbone` already does this deliberately
   ("Benchmark answers must be direct"), and the quoted Tinker reference rows were produced that
   way, so this aligns the local path with the harness's existing rule. Applied identically to
   base and LoRA, so the control is preserved. This is a change to `gold_eval/backbones.py`, not
   to any prompt.
4. **Single seed only.** Seeds 1 and 2 and the INDEP-only / JOINT-only ablations (PLAN §4,
   step 2) were **not run** — the main run alone took 3.2 h training + ~2 h eval. Reported as
   single-run, flagged **C9** per PLAN §4. No mean±std is available.
5. **T2 (match) and T3 (rationale) collateral rows were not run** (PLAN §5.3). Out of scope of
   this run's instructions, and T2 is in any case blocked pending the option-set rebuild
   (EVIDENCE C12).
6. **A post-hoc tie-aware sensitivity analysis was added** (`sensitivity_tied_as_zero` in
   `finetune_paired.json`). It is clearly labelled as post-hoc and does not replace the
   pre-registered statistic, but the tie-rate asymmetry it exposes is large enough that
   reporting the pre-registered number alone would mislead.

### Code fixes required to make the unrun node path execute

7. **`SFTData` discarded all but the first image patch.** `Qwen3VLProcessor` returns
   `pixel_values` as a flat `(total_patches, dim)` tensor and `image_grid_thw` as `(n_img, 3)` —
   neither carries a batch axis — so `{k: v[0] for k, v in enc.items()}` truncated them to a
   single patch and a single image's grid. Training would have run on ~1/32 of each video
   without erroring. Now the processor's own batching is kept and `collate` is a passthrough.
8. **Assistant mask was off by two.** With the default template, `prompt_text` is a *string*
   prefix of `full_text` but not a *token* prefix (the `<think>` boundary retokenises), so
   `labels[:n_prompt] = -100` cut in the wrong place and left `</think>\n\n` in the loss.
   Rendering the training prompt with `enable_thinking=False` makes it an exact token prefix —
   and matches the eval-time rendering from (3).
9. **`warmup_ratio` → `warmup_steps=0.03`.** transformers v5 removed `warmup_ratio`;
   `get_warmup_steps()` reads a `warmup_steps` float < 1 as a fraction of total steps, so
   PLAN §4's "warmup 3%" is preserved exactly.
10. **8-rank launch fixed.** Eight ranks each materialised a 19 GB CPU copy of the weights
    before anything touched CUDA; the first CUDA call then failed with `CUDA driver
    initialization failed`, which `TrainingArguments` surfaced as the misleading "Your setup
    doesn't support bf16/gpu". `TrainingArguments` is now built first (initialising the
    accelerator while host memory is free) and each rank loads with
    `device_map={"": LOCAL_RANK}`. A single-entry device_map leaves `Trainer.is_model_parallel`
    False, so this is still plain DDP as PLAN §4 specifies.
11. **`save_pretrained` guarded on rank 0** — all 8 ranks previously wrote the same adapter
    directory concurrently. `train_args.json` now also records `trainable_params`,
    `global_step`, `best_eval_loss`, `best_eval_step` and the full `log_history`.

### Environmental / cosmetic

12. `gold_eval/data` and `finetune/data` are **symlinks to `/srv/local/scratch/yiboz7/rvb/`** —
    the root filesystem was at 98% (62 GB free) and the clips+frames cache reached 5.1 GB
    alongside a 19 GB model download. `.gitignore` gained matching entries, because a symlink is
    a file and the existing `finetune/data/` rule did not cover `finetune/data`.
13. `tinker` / `tinker-cookbook` were **not** installed — this run used only the local HF path,
    and they would have pulled unrelated dependencies.
14. `summarize.py` rebuilds `summary.json` from scratch each run. Verified before overwriting
    that every pre-existing row (gemini / gpt-5 / all tinker arms) reproduces bit-identically,
    with only the four new `hf_` rows added.

### Observed, left untouched

15. **Train/eval prompt-ordering asymmetry.** PLAN §3 specifies the training user turn as
    `[frames] + prompt`, whereas the eval runners send `prompt → frames → "Score (0-100):"`.
    So the fine-tune trains on a different part-ordering than it is evaluated under. This is a
    pre-registered design decision and I did not change it, but it is a plausible dampener on
    transfer and a candidate first fix for run 2.
16. **`--indep_cap` is inert.** INDEP windows are ~3 s at 1 fps and yield 2–4 frames
    (median 3), so the 12-frame cap never binds. INDEP inputs are therefore *identical* between
    the cap-32 and cap-24 builds, which is why deviation (1) cannot explain the INDEP result.
