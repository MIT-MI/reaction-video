# LoRA fine-tune run 2 / R2 — intensity, order-aligned, 5 epochs (Qwen3.5-9B)

Executed 2026-08-24 against `finetune/PLAN.md` and the R2 batch spec. Every number below comes
from a file under `gold_eval/results/` or `finetune/ckpt/r2-order-5ep/train_args.json`; nothing
is back-filled or estimated. Run-1 numbers quoted for comparison come from `finetune/REPORT.md`
and `finetune/ckpt/Qwen_Qwen3.5-9B-lora-r16/train_args.json`.

**R2 = run 1 with exactly two changes:** (a) training samples use the EVAL part order
(text prompt → frames → short cue), fixing run 1's Deviation #15; (b) 5 epochs instead of 3,
because run 1's val loss was still falling at its final step.

---

## Setup

Identical to run 1 — same node, same venv, same package versions, same base-model revision — so
the with/without contrast carries no environment confound. Repeated here for self-containment:

| | |
|---|---|
| Node | `ise-dynamo.cs.illinois.edu`, Linux 5.15.0-161-generic, 96 vCPU, 503 GB RAM |
| GPUs | 8 × NVIDIA RTX A6000, 49,140 MiB each, driver 575.57.08 (CUDA 12.9) |
| Python / torch | 3.13.13 / 2.11.0+cu128 |
| transformers / peft / accelerate | 5.15.1 / 0.20.0 / 1.14.0 |
| numpy / scipy / pillow | 2.5.2 / 1.18.1 / 12.3.0 |
| Base model | `Qwen/Qwen3.5-9B` |
| Code revision | `reaction-video` @ `5454785` (gold-eval), dataset repo @ `2180086` |

No loader edits were needed (as in run 1). No new deviations of the run-1 kind were required —
the code fixes enumerated in REPORT.md §Deviations 7–11 were already in the tree.

---

## Data

`.venv/bin/python -m finetune.build_sft_data --split .../reports/phase0/finetune --joint_cap 24`

| split | INDEP rows | JOINT rows (oversampled ×2 on train) | total | failed videos |
|---|---|---|---|---|
| train | 1,952 | 856 (from 428 multi-moment videos) | 2,808 | **0** |
| val | 213 | 48 (×1) | 261 | **0** |

Exactly the counts the spec required and exactly run 1's counts. All 3,069 rows validated: every
`images` path exists on disk, every INDEP answer is an integer in 0–100, every JOINT answer parses
as JSON with `m0..mk` integer keys in 0–100, and every row carries `prompt_before`.

**Change (a) verified at the string level, not just the field level.** Each row's `prompt_before`
was asserted equal to the prompt object imported from the eval runner
(`run_ranking_independent.PROMPT` for INDEP; the `run_ranking.PROMPT` prefix for JOINT), so train
and eval strings still cannot drift. The rendered training prompt now ends:

```
...Reply with a single integer only.<|vision_start|><|image_pad|><|vision_end|>×N Score (0-100):
```

i.e. prompt → frames → cue, which is byte-for-byte the layout
`run_ranking_independent.moment_parts` and `run_ranking.build_parts` send at eval time
(`[{"text": PROMPT}] + images + [{"text": "Score (0-100):" / "JSON scores:"}]`).

**Token lengths** (120 sampled rows per split×format, tokenised through the real `SFTData`):

| split | format | min | p50 | p90 | max | mean | run-1 mean |
|---|---|---|---|---|---|---|---|
| train | indep | 613 | 871 | 1,129 | 1,129 | 946 | 947 |
| train | joint | 4,600 | 6,409 | 6,434 | 6,484 | 6,399 | 6,400 |
| val | indep | 612 | 871 | 1,129 | 1,129 | 946 | 947 |
| val | joint | 5,117 | 6,409 | 6,434 | 6,457 | 6,386 | 6,388 |

**0 samples exceeded the 10,000-token skip threshold**, so no training row was silently dropped.
Reordering the parts left sequence length essentially unchanged (≤1 token of mean drift), so R2's
compute per epoch is run 1's — the reorder is a pure part-order intervention, not a length one.

Rebuild took ~15 s: run 1's clip and frame caches were reused, nothing was re-downloaded and no
frame was re-extracted, so the *pixels* are identical to run 1's. JOINT frame cap stays 24
(run 1 Deviation #1), so the ≤24-train / ≤64-eval length extrapolation still applies.

---

## Training

`torchrun --nproc_per_node 8 -m finetune.train_lora --base Qwen/Qwen3.5-9B --out finetune/ckpt/r2-order-5ep --epochs 5 --lr 1e-4 --rank 16 --seed 0`

| | R2 | run 1 |
|---|---|---|
| Trainable params | **29,097,984** (0.3083% of 9,438,911,728) | 29,097,984 |
| Optimizer steps | 440 (5 epochs, 88/epoch) | 264 (3 epochs) |
| Effective batch | 32 (8 GPUs × 1 × grad-accum 4) | 32 |
| LR schedule | cosine, warmup 3%, peak 1e-4 | same |
| Wall-clock | **18,986.8 s = 5.27 h** | 11,430 s = 3.18 h |
| Train loss | 1.0046 (first logged) → 0.2157 (last); run mean 0.3913 | 0.967 → 0.365; mean 0.4635 |
| **Best val loss** | **0.39611 @ step 300** | 0.39549 @ step 264 (final) |

Trainable-parameter count is bit-identical to run 1 (the hybrid-attention structural explanation
in REPORT.md §Training still applies). Loss masking verified again under the new part order —
the unmasked span is exactly the answer plus the turn terminator:

```
indep row  seq_len=871   unmasked=4   decoded='33<|im_end|>\n'                answer='33'
indep row  seq_len=871   unmasked=4   decoded='42<|im_end|>\n'                answer='42'
joint row  seq_len=2793  unmasked=18  decoded='{"m0": 60, "m1": 60}<|im_end|>\n'
```

### Change (b) answered: 5 epochs over-trains

Full val-loss trace (`train_args.json` → `log_history`):

| step | 100 | 200 | 300 | 400 | 440 |
|---|---|---|---|---|---|
| epoch | 1.14 | 2.27 | 3.41 | 4.55 | 5.00 |
| **R2 val loss** | 0.4461 | 0.4275 | **0.39611 (best)** | 0.4920 | 0.4939 |
| run-1 val loss | 0.46458 | 0.40885 | — | — | 0.39549 @ 264 |

Run 1 inferred under-training from a val loss still falling at step 264. R2 shows that reading was
wrong: the minimum sits at **step 300 (~3.4 epochs)** at 0.39611 — statistically indistinguishable
from run 1's 0.39549 — after which val loss rises sharply (+0.098 by step 440) while train loss
keeps dropping to 0.2157. That is textbook over-fitting, and it means **run 1 was already at the
val-loss floor.** The extra epochs bought nothing.

`load_best_model_at_end=True` therefore selected the step-300 checkpoint, and
`finetune/ckpt/r2-order-5ep/best` **is the step-300 adapter, not the 5-epoch endpoint** — every
result below is that checkpoint. (Trainer protects `best_model_checkpoint` from `save_total_limit`
rotation; checkpoint-300 was verified present on disk at the end of the run.) `EarlyStoppingCallback`
(patience 3) saw only two worsening evaluations and so never fired.

Note the two changes are therefore **not** cleanly separable on val loss: R2's best checkpoint is at
3.4 epochs, so what is evaluated below is "order-aligned, early-stopped at ~3.4 epochs" versus run 1's
"misordered, 3 epochs". The 5-epoch instruction was carried out as specified; it selected itself out.

---

## Results

Frozen 250-video ranking set, frames protocol untouched (JOINT ≤64 frames, INDEP window ≤12), both
arms through `HFLocalBackbone` with greedy decoding. **Base rows were reused from run 1, not
re-run**, as instructed. Unparsed responses: **0** in both new runs (250/250 scored on both tasks).
ρ against the rater-debiased consensus equals ρ against the raw mean to 4 dp for both new runs
(`mean_spearman_debiased == mean_spearman` in `summary.json`), as in run 1.

### Per-arm (source: `gold_eval/results/summary.json`)

Two conventions are shown per task. **tie-aware** credits ρ = 0 to an all-tied (unrankable)
prediction over all 250 videos; **drop-ties** is the run-1 pre-registered scorer, which discards
those videos. See the collapse table for why the difference matters.

| arm | T1-JOINT ρ tie-aware | T1-JOINT ρ drop-ties [95% CI] (n) | T1-INDEP ρ tie-aware | T1-INDEP ρ drop-ties [95% CI] (n) |
|---|---|---|---|---|
| base, local path | 0.0236 | 0.0236 [−0.0862, 0.1344] (250) | 0.2534 | 0.3184 [0.1998, 0.4334] (199) |
| run-1 LoRA (3 ep, misordered) | 0.0857 | 0.1291 [−0.0077, 0.2620] (166) | 0.3423 | 0.5217 [0.4123, 0.6281] (164) |
| **R2 LoRA (order-aligned)** | **0.1352** | **0.2964 [0.1428, 0.4407] (114)** | **0.3730** | **0.5239 [0.4191, 0.6207] (178)** |
| *Tinker zero-shot 9B (ref)* | — | *0.0294 [−0.0799, 0.1366] (250)* | — | *0.3340 [0.2160, 0.4482] (191)* |
| *Human LOO ceiling (ref)* | *0.649* | *0.649* | *0.649* | *0.649* |

### Paired test — PRIMARY: tie-aware (source: `gold_eval/results/finetune_paired_r2.json`)

`per_video_rhos(..., tied_as_zero=True)` over all 250 videos (all 250 have non-tied gold; zero
videos required missing-row imputation on either arm), Wilcoxon signed-rank + seeded percentile
bootstrap CI (`RNG_SEED=0`, `N_BOOT=10000`).

| task | n | base ρ | R2 ρ | **Δρ** | Δ 95% CI | Wilcoxon p | improved/worsened/tied | run-1 Δρ (p) |
|---|---|---|---|---|---|---|---|---|
| **T1-JOINT** | 250 | 0.0236 | 0.1352 | **+0.1116** | [−0.0139, 0.2326] | **0.171** | 109 / 89 / 52 | +0.0621 (0.320) |
| T1-INDEP | 250 | 0.2534 | 0.3730 | **+0.1196** | [0.0168, 0.2202] | **0.0516** | 92 / 72 / 86 | +0.0888 (0.117) |

> **Pre-registered criterion — Δρ_JOINT ≥ 0.10 with p < 0.05, judged under the primary
> (tie-aware) convention: NOT MET.** On this seed the effect-size bar is cleared (+0.112, up from
> run 1's +0.062) but the significance bar is not (p = 0.171).
> **Superseded by the seed replicates below:** seed 0 is the most favourable of three seeds, and
> the across-seed mean (+0.081 ± 0.030) does **not** clear the effect-size bar either.

### Paired test — SECONDARY: pre-registered drop-ties convention

| task | n paired | base ρ | R2 ρ | Δρ | Δ 95% CI | Wilcoxon p | improved/worsened/tied |
|---|---|---|---|---|---|---|---|
| **T1-JOINT** | 114 | 0.0959 | 0.2964 | **+0.2005** | [0.0039, 0.3897] | **0.0348** | 42 / 26 / 46 |
| T1-INDEP | 143 | 0.3878 | 0.5120 | +0.1242 | [−0.0085, 0.2557] | 0.133 | 40 / 33 / 70 |

**Under the drop-ties convention the pre-registered criterion would read as MET on T1-JOINT
(Δ = +0.20, p = 0.035).** It is not met under the primary convention. This divergence is the single
most important result of R2 and is explained by the next table: the drop-ties number is computed on
the 114 videos the LoRA did not collapse on, having silently deleted the 136 it did.

### Output-range collapse (source: `finetune_paired_r2.json` → `coverage`)

| task | arm | all-tied (unrankable) | prediction SD | pred mean | distinct values | n scored (drop-ties) |
|---|---|---|---|---|---|---|
| T1-JOINT | base | 0 / 250 | 27.89 | 55.20 | 28 | 250 |
| T1-JOINT | run-1 LoRA | 84 / 250 | 9.81 | — | 12 | 166 |
| T1-JOINT | **R2 LoRA** | **136 / 250** | 13.26 | 60.40 | 13 | **114** |
| T1-INDEP | base | 51 / 250 | 22.97 | 61.16 | 16 | 199 |
| T1-INDEP | run-1 LoRA | 86 / 250 | 10.80 | — | 10 | 164 |
| T1-INDEP | **R2 LoRA** | **72 / 250** | 12.79 | 57.62 | 20 | **178** |

Output-range collapse **did not go away, and on the joint task it got worse**: R2 emits identical
scores for every moment on 136/250 joint videos, against 84 for run 1 and 0 for the base. Prediction
SD remains roughly half the base's (13.26 vs 27.89). On the independent task collapse eased slightly
(86 → 72 all-tied, 10 → 20 distinct values). So the order alignment made the model *slightly* more
willing to spread its independent scores and *less* willing to spread its joint ones.

### Did the calibration gap close? (source: `summary.json` → `ranking_paired`)

E3's binding failure is the within-model gap between independent and joint scoring. Under the
tie-aware convention, paired over all 250 videos:

| arm | joint ρ | indep ρ | **gap** | Wilcoxon p |
|---|---|---|---|---|
| base | 0.0236 | 0.2534 | **0.2298** | 0.0012 |
| run-1 LoRA | 0.0857 | 0.3423 | 0.2565 | 2.2e-05 |
| **R2 LoRA** | 0.1352 | 0.3730 | **0.2379** | 1.6e-05 |

The gap is **0.2298 for the base and 0.2379 after fine-tuning** — unchanged within noise, and still
overwhelmingly significant. Both arms rose together; the joint-versus-independent deficit did not
narrow.

---

## Seed replicates (R6): seeds 1 and 2

Added 2026-08-24 after R5, per the R6 spec (batch-2 `match_train.json` was absent at launch, gate
re-checked against a fresh `git fetch`). Seeds 1 and 2 are R2 re-run verbatim with `--seed 1` /
`--seed 2` → `finetune/ckpt/r2-seed1`, `finetune/ckpt/r2-seed2`. Trainable params 29,097,984 in
both; wall-clock 5.26 h and 5.29 h.

| seed | best val loss @ step | wall-clock |
|---|---|---|
| 0 (`r2-order-5ep`) | 0.39611 @ 300 | 5.27 h |
| 1 (`r2-seed1`) | 0.42641 @ 200 | 5.26 h |
| 2 (`r2-seed2`) | 0.43449 @ 200 | 5.29 h |

All three overfit; seeds 1 and 2 bottom out a full 100 steps earlier than seed 0 and at a
noticeably worse value, so even the *location* of the optimum is seed-dependent.

### Per-seed results, PRIMARY tie-aware (n = 250 for every row)

| seed | T1-JOINT ρ | Δ vs base [95% CI] | p | T1-INDEP ρ | Δ vs base [95% CI] | p |
|---|---|---|---|---|---|---|
| 0 | 0.1352 | +0.1116 [−0.0139, 0.2326] | 0.171 | 0.3730 | +0.1196 [0.0168, 0.2202] | 0.0516 |
| 1 | 0.1033 | +0.0797 [−0.0241, 0.1800] | 0.227 | 0.3615 | +0.1080 [0.0119, 0.2031] | 0.0509 |
| 2 | 0.0753 | +0.0517 [−0.0627, 0.1641] | 0.514 | 0.3373 | +0.0839 [−0.0139, 0.1832] | 0.136 |
| **mean ± sd** | **0.1046 ± 0.0300** | **+0.0810 ± 0.0300** | — | **0.3573 ± 0.0182** | **+0.1038 ± 0.0182** | — |

**Seed 0 is the most favourable of the three on both tasks.** The across-seed mean Δρ_JOINT is
**+0.081 ± 0.030, which does not clear the pre-registered ≥0.10 effect-size bar**, and no individual
seed reaches p < 0.05 on either task (JOINT p = .171/.227/.514; INDEP p = .052/.051/.136). The
single-seed R2 headline of +0.112 sits at the top of the seed distribution and **must not be quoted
as the effect size.** C9 is thereby discharged for R2: the result is a null under replication, not
merely under-powered on one run.

### Collapse and the drop-ties artefact across seeds

| seed | JOINT all-tied | pred SD | n scored (drop-ties) | drop-ties Δρ_JOINT | p |
|---|---|---|---|---|---|
| 0 | 136 / 250 | 13.26 | 114 | +0.2005 | **0.0348** |
| 1 | 126 / 250 | 10.65 | 124 | +0.0944 | 0.155 |
| 2 | **193 / 250** | 11.94 | **57** | **+0.2350** | 0.125 |
| mean ± sd | 151.7 ± 36.1 | 11.95 ± 1.31 | — | — | — |

Collapse is severe and highly variable (126–193 of 250 joint videos, sd = 36), and this table is the
cleanest demonstration in the batch of why the convention had to change: **the most collapsed seed
produces the largest drop-ties improvement.** Seed 2 ties on 193/250 videos, leaving the scorer just
57 to average over, and on those 57 it posts the batch's biggest drop-ties delta (+0.235) — while
the tie-aware convention, which sees all 250, ranks it the *worst* of the three (+0.052). Under
drop-ties the three seeds would be reported as +0.201 / +0.094 / +0.235; under tie-aware they are
+0.112 / +0.080 / +0.052, ordered by how much the model actually differentiated.

### E3 gap across seeds (tie-aware)

| arm | joint ρ | indep ρ | gap | p |
|---|---|---|---|---|
| base | 0.0236 | 0.2534 | 0.2298 | 0.0012 |
| seed 0 | 0.1352 | 0.3730 | 0.2379 | 1.6e-05 |
| seed 1 | 0.1033 | 0.3615 | 0.2582 | 6.0e-06 |
| seed 2 | 0.0753 | 0.3373 | 0.2620 | <1e-06 |

Every seed leaves the independent-vs-joint gap **wider** than the base's 0.2298, and overwhelmingly
significant in all three. No seed shows any sign of calibration repair.

---

## Honest interpretation

> **Revised after the R6 seed replicates.** The paragraphs below were written from seed 0 alone.
> The single-seed conclusion (null, but with the effect-size bar cleared) survives only in its
> "null" half: across three seeds the mean Δρ_JOINT is +0.081 ± 0.030 and the effect-size bar is
> **not** cleared either. Read the seed section above as the authoritative result for R2.

Fixing the part-order asymmetry and training longer **moved every number in the right direction and
still did not meet the pre-registered criterion.** Under the primary tie-aware convention T1-JOINT
improved by Δρ = +0.112 (run 1: +0.062), clearing the ≥0.10 effect bar, but at p = 0.171 over 250
paired videos it remains indistinguishable from zero; T1-INDEP improved by +0.120 and lands at
p = 0.052, just the wrong side of the line. So R2 is **again a null result for the calibration
hypothesis at this scale** — and the seed replicates make it a firmer null: across seeds 0–2 the
joint effect is +0.081 ± 0.030, below the effect-size bar, with seed 0 the most favourable draw.

Two findings from R2 matter more than its headline deltas.

**1. The scoring convention decides the verdict, and the honest one says no.** The drop-ties
statistic reports Δρ_JOINT = +0.20 at p = 0.035 — the pre-registered criterion, met. The tie-aware
statistic reports +0.112 at p = 0.171 — not met. Both describe the same predictions. The difference
is that R2 collapsed to an unrankable all-tied output on 136 of 250 joint videos, and the drop-ties
scorer removes exactly those videos from the fine-tuned arm's own average, scoring it only where it
chose to differentiate while the base is scored on all 250. Had run 1's pre-registered convention
been carried into R2 unmodified, this batch would have reported a positive, significant result for
the fine-tune. Making tie-aware primary is what prevents that, and it should be treated as the
headline methodological outcome of this run rather than a footnote.

**2. Run 1's under-training diagnosis was wrong.** Val loss bottoms out at 0.39611 at ~3.4 epochs,
against run 1's 0.39549 at 3.0, then degrades sharply. Run 1's "still falling at the final step"
reading was noise at the floor, not a trend. Combined with (1), the two run-2 interventions did what
they could: the *ordering* fix is real and plausibly explains the improved deltas, but there was no
under-training headroom for the *epochs* fix to exploit, and neither addressed the actual failure
mode, which is that supervised exposure to human-anchored labels teaches this model to predict
something near the label mean rather than to rank moments within a video. The
independent-versus-joint gap that motivated the whole experiment is 0.2298 at base and 0.2379 after
fine-tuning — untouched.

The eye-catching drop-ties numbers (.0236 → .2964 joint, .3184 → .5239 indep) must not be quoted
without their n (114, 178) and the collapse counts that produced them.

---

## Exact commands run

```bash
cd reaction-video && git checkout gold-eval && git pull      # @5454785
cd ../reaction_video_benchmark_phase0_pilot && git pull      # @2180086

# 1. data (clip/frame caches from run 1 reused; ~15 s)
.venv/bin/python -m finetune.build_sft_data \
    --split ../reaction_video_benchmark_phase0_pilot/reports/phase0/finetune --joint_cap 24

# 2. training (8 × A6000, 5.27 h)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/torchrun --nproc_per_node 8 -m finetune.train_lora \
    --base Qwen/Qwen3.5-9B --train finetune/data/train.jsonl --val finetune/data/val.jsonl \
    --out finetune/ckpt/r2-order-5ep --epochs 5 --lr 1e-4 --rank 16 --seed 0

# 3. eval — LoRA only; base rows reused from run 1. Run concurrently on 4 GPUs each.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3 \
.venv/bin/python -m gold_eval.run_ranking             --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r2-order-5ep/best"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4,5,6,7 \
.venv/bin/python -m gold_eval.run_ranking_independent --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r2-order-5ep/best"

# 4. paired test (tie-aware primary) + summary
.venv/bin/python -m finetune.paired_test --lora finetune/ckpt/r2-order-5ep/best \
    --out gold_eval/results/finetune_paired_r2.json
.venv/bin/python -m gold_eval.summarize
```

Wall-clock: data ~15 s; training 5.27 h; eval 32 min (JOINT) and 8 min (INDEP) run concurrently
→ 32 min; paired test + summarize < 1 min. **Total ≈ 5.9 h.**

### R6 seed replicates

```bash
# gate re-checked first: reports/phase0/finetune/match_train.json absent after a fresh git fetch
for S in 1 2; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/torchrun --nproc_per_node 8 -m finetune.train_lora --base Qwen/Qwen3.5-9B \
      --train finetune/data/train.jsonl --val finetune/data/val.jsonl \
      --out finetune/ckpt/r2-seed$S --epochs 5 --lr 1e-4 --rank 16 --seed $S
done
# evals: round 1 = both seeds' T1-JOINT, round 2 = both seeds' T1-INDEP (4 GPUs each)
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m gold_eval.run_ranking             --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r2-seed1/best"
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python -m gold_eval.run_ranking             --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r2-seed2/best"
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m gold_eval.run_ranking_independent --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r2-seed1/best"
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python -m gold_eval.run_ranking_independent --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r2-seed2/best"
.venv/bin/python -m finetune.paired_test --lora finetune/ckpt/r2-seed1/best \
    --out gold_eval/results/finetune_paired_r2_seed1.json
.venv/bin/python -m finetune.paired_test --lora finetune/ckpt/r2-seed2/best \
    --out gold_eval/results/finetune_paired_r2_seed2.json
.venv/bin/python -m gold_eval.summarize
```

R6 wall-clock: 5.26 h + 5.29 h training, 40 min evals → **≈ 11.2 h.**

---

## Deviations from the R2 spec

**Nothing under `gold_eval/tasks/` was touched. No evaluation prompt string and no frame protocol
was altered.** Prompts are still imported from the eval runners by `build_sft_data.py`.

1. **The evaluated checkpoint is step 300 (~3.4 epochs), not the 5-epoch endpoint.** This is
   `load_best_model_at_end` + `metric_for_best_model="eval_loss"` behaving exactly as configured in
   run 1, not a change — but since the spec's change (b) was "5 epochs", it must be stated plainly
   that 5 epochs of *optimisation* were run while the *selected* adapter is the early-stopped one.
   Training the full 5 epochs and evaluating the endpoint would have evaluated a checkpoint 0.098
   worse in val loss. Recorded in `train_args.json` as `best_eval_step: 300`.
2. **`finetune/paired_test.py` was rewritten** so the tie-aware convention is primary, per spec
   step 4. The pre-registered drop-ties statistic is retained verbatim alongside it. Verified by
   re-running the new code against run 1's checkpoint: it reproduces every published run-1 number
   exactly (drop-ties JOINT Δ+0.1052 / p 0.13119 / n 166; INDEP Δ+0.1844 / p 0.003074 / n 136;
   tie-aware +0.0621 / +0.0888; SD 27.89 → 9.81 and 22.97 → 10.80; all-tied 0/84 and 51/86). The
   rewrite therefore changed presentation and defaults, not statistics. Output now goes to a
   per-run file (`finetune_paired_r2.json`) so run 1's `finetune_paired.json` is preserved.
3. **`finetune/split_by_format.py` added** for R5 (spec's "short python script"), committed for
   reproducibility rather than run ad hoc.
4. **Base T1 rows were reused from run 1, not re-run**, as instructed.
5. **T2 (match) / T3 (rationale) collateral rows not run** — out of scope for this batch, and T2
   remains blocked pending the option-set rebuild (EVIDENCE C12).
6. **Seeds 1 and 2 were run (R6)** after R5, once the batch-2 gate was re-checked and still open;
   see the seed-replicates section. C9 is discharged for R2. The R5 ablations remain single-seed.
8. **The interpretation was revised after R6.** The Results and Honest-interpretation sections were
   written from seed 0 before seeds 1–2 existed; rather than rewrite them, the seed section was
   added and the two places whose conclusion changed are marked inline. No seed-0 number was
   altered.
7. `finetune/data/{train,val}_run1.jsonl.bak` hold run 1's exact SFT files, since the rebuild
   overwrote `train.jsonl` / `val.jsonl` in place. Untracked (the `finetune/data` symlink is
   gitignored per run-1 Deviation #12).
