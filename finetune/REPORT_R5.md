# R5 — single-format ablations (INDEP-only vs JOINT-only), 5 epochs

Executed 2026-08-24 against the R5 batch spec, immediately after R2. Every number comes from a file
under `gold_eval/results/` or a checkpoint's `train_args.json`. R2 and base numbers are reproduced
from `finetune/REPORT_R2.md` / `summary.json`; run-1 numbers from `finetune/REPORT.md`.

**Question:** which of the two training formats drives what R2 showed? Everything is held at R2 —
same rows, same labels, same eval-aligned part order, same joint ×2 oversampling, same
hyperparameters, seed 0 — and only the *format mixture* changes.

---

## Data

`finetune/split_by_format.py` filters the R2 files by the `format` field. No row is rebuilt,
re-cut or re-labelled, so the ablation arms see byte-identical samples to the ones R2 trained on.

| file | rows | | file | rows |
|---|---|---|---|---|
| `train_indep.jsonl` | 1,952 | | `train_joint.jsonl` | 856 (×2 oversampled, as in R2) |
| `val_indep.jsonl` | 213 | | `val_joint.jsonl` | 48 |

1,952 + 856 = 2,808 = R2's train total, and 213 + 48 = 261 = R2's val total, so the split is exact
and exhaustive. Every row carries `prompt_before` (verified).

---

## Training

Both: `torchrun --nproc_per_node 8 -m finetune.train_lora --base Qwen/Qwen3.5-9B --epochs 5 --lr 1e-4 --rank 16 --seed 0`.
Trainable params **29,097,984** in both, identical to R2 and run 1.

| run | train rows | steps (per epoch) | wall-clock | val-loss trace | best |
|---|---|---|---|---|---|
| R2 (mixture) | 2,808 | 440 (88) | 5.27 h | .4461 / .4275 / **.39611** / .4920 / .4939 | step 300 (~3.4 ep) |
| **r5-indep** | 1,952 | 305 (61) | **0.65 h** | .52909 / **.47796** / .49871 / .49711 | step 200 (~3.3 ep) |
| **r5-joint** | 856 | 135 (27) | **2.35 h** | **.24798** / .31081 | step 100 (~3.7 ep) |

**Val losses are not comparable across rows** — each arm is evaluated on its own format's val
subset, so r5-joint's .248 against r5-indep's .478 reflects how much easier next-token prediction
is on a short JSON answer than on a bare integer, not model quality. Within each row the trace is
what matters, and all three tell the same story: **the minimum is at ~3.3–3.7 epochs and the 5th
epoch is actively harmful.** In all three runs `load_best_model_at_end` therefore evaluated an
earlier checkpoint, not the 5-epoch endpoint.

**Caveat on r5-joint's checkpoint selection.** With 135 total steps and `eval_steps=100` held at
R2's value, r5-joint got only two validation evaluations (steps 100 and 135). Its "best" is
selected from two candidates, far coarser than r5-indep's four or R2's five. Holding `eval_steps`
at R2 was required by "all else = R2", so this was not changed, but r5-joint's checkpoint is
plausibly further from its true optimum than the others'.

---

## Results

Frozen 250-video ranking set, frames protocol untouched, `HFLocalBackbone` greedy. Base rows reused
from run 1. **Unparsed responses: 0 in all four new runs** (250/250 videos scored on both tasks by
both arms). ρ vs debiased consensus equals ρ vs raw mean to 4 dp for every arm.

### The 4-row table — PRIMARY tie-aware convention

ρ = `per_video_rhos(tied_as_zero=True)` over all 250 videos; Δ, CI and p are paired vs the local
base (`finetune_paired_r5_{indep,joint}.json`, `finetune_paired_r2.json`).

| arm | T1-JOINT ρ | Δ vs base [95% CI] | p | T1-INDEP ρ | Δ vs base [95% CI] | p |
|---|---|---|---|---|---|---|
| base (local) | 0.0236 | — | — | 0.2534 | — | — |
| R2 (INDEP+JOINT) | 0.1352 | +0.1116 [−0.0139, 0.2326] | 0.171 | 0.3730 | +0.1196 [0.0168, 0.2202] | 0.0516 |
| **r5-indep** (INDEP only) | 0.1044 | +0.0808 [−0.0086, 0.1701] | 0.0801 | **0.3728** | **+0.1193 [0.0247, 0.2157]** | **0.0195** |
| **r5-joint** (JOINT only) | **0.1441** | **+0.1205 [0.0041, 0.2397]** | 0.0776 | 0.2114 | **−0.0420** [−0.1430, 0.0621] | 0.401 |

> **Pre-registered criterion (Δρ_JOINT ≥ 0.10, p < .05) is not met by any arm.** r5-joint clears
> the effect bar (+0.121) with p = 0.078; R2 clears it (+0.112) with p = 0.171.
> The one result in this batch that is significant under the primary convention is **r5-indep on
> T1-INDEP (+0.119, p = 0.0195)** — the axis that was already strong.

### Secondary: pre-registered drop-ties convention

| arm | T1-JOINT n | Δρ | p | T1-INDEP n | Δρ | p |
|---|---|---|---|---|---|---|
| R2 | 114 | +0.2005 | **0.0348** | 143 | +0.1242 | 0.133 |
| r5-indep | 246 | +0.0745 | 0.108 | 148 | +0.1304 | **0.0345** |
| r5-joint | 136 | +0.1136 | 0.175 | 104 | +0.0018 | 0.961 |

Note how the convention reshuffles the ranking: drop-ties makes **R2** look like the best joint arm
(+0.20, significant) when tie-aware makes **r5-joint** the best (+0.121, n.s.) and shows R2's
advantage was largely denominator shrinkage. R2's drop-ties JOINT number rests on 114 videos;
r5-indep's rests on 246.

### Output-range collapse — the ablation's clearest finding

| task | arm | all-tied (unrankable) | prediction SD | distinct values | n scored (drop-ties) |
|---|---|---|---|---|---|
| T1-JOINT | base | 0 / 250 | 27.89 | 28 | 250 |
| T1-JOINT | R2 | 136 / 250 | 13.26 | 13 | 114 |
| T1-JOINT | **r5-indep** | **4 / 250** | 13.75 | 15 | **246** |
| T1-JOINT | r5-joint | 114 / 250 | 10.99 | 11 | 136 |
| T1-INDEP | base | 51 / 250 | 22.97 | 16 | 199 |
| T1-INDEP | R2 | 72 / 250 | 12.79 | 20 | 178 |
| T1-INDEP | **r5-indep** | 72 / 250 | 10.91 | 17 | 178 |
| T1-INDEP | r5-joint | **131 / 250** | 8.32 | 12 | 119 |

**The JOINT training format is what causes the joint-task collapse.** Trained on INDEP samples
only, the model emits all-tied joint predictions on just **4 of 250** videos — essentially the
base's behaviour (0) and nothing like R2's 136 or joint-only's 114 — while still improving joint ρ
by +0.081. Since r5-indep never sees a joint prompt in training, this also means the joint gain it
does achieve is pure transfer from per-moment supervision.

Conversely, JOINT-only training collapses *both* tasks (114 and 131 all-tied) and is the only arm
that drives independent-task SD below 10.

### Format specificity and transfer

| training format | effect on T1-JOINT | effect on T1-INDEP |
|---|---|---|
| INDEP only | +0.0808 (transfer) | +0.1193 (own format) |
| JOINT only | +0.1205 (own format) | **−0.0420 (negative transfer)** |
| both (R2) | +0.1116 | +0.1196 |

Each format's gain is essentially its own: R2's joint gain (+0.112) is close to joint-only's
(+0.121), and R2's indep gain (+0.120) is nearly identical to indep-only's (+0.119) — the mixture
adds nothing measurable beyond the sum of its parts, and on T1-INDEP R2 and r5-indep are
indistinguishable (ρ 0.3730 vs 0.3728, both 72 all-tied, both n = 178). What the mixture *does*
inherit from its joint half is the collapse: 136 all-tied joint predictions versus 4 for indep-only.
Meanwhile joint-only actively damages independent scoring (−0.042).

### Did any arm close the E3 gap? (`summary.json` → `ranking_paired`, tie0)

| arm | joint ρ | indep ρ | gap | p |
|---|---|---|---|---|
| base | 0.0236 | 0.2534 | 0.2298 | 0.0012 |
| R2 | 0.1352 | 0.3730 | 0.2379 | 1.6e-05 |
| r5-indep | 0.1044 | 0.3728 | **0.2684** | 5.4e-05 |
| **r5-joint** | 0.1441 | 0.2114 | **0.0673** | 0.286 |

r5-joint is the only arm whose independent-vs-joint gap is no longer significant — but it gets there
**by pulling independent scoring down** (0.2534 → 0.2114) rather than by lifting joint scoring to
meet it. That is the gap closing for the wrong reason and must not be reported as calibration
repair. r5-indep *widens* the gap the most (0.2684) precisely because it improves the independent
axis while barely moving joint.

---

## Honest interpretation

The ablation answers R5's question cleanly: **the two formats do different things, and neither one
fixes calibration.** JOINT-format supervision is what moves the joint metric (+0.121 vs indep-only's
+0.081) — and it is also what teaches the model to stop differentiating, collapsing 114/250 joint
videos to a single repeated score and inflicting −0.042 of negative transfer on the independent
task. INDEP-format supervision produces the batch's only result that is significant under the
primary convention (+0.119 on T1-INDEP, p = 0.0195), transfers +0.081 to the joint task, and leaves
the model's willingness to differentiate almost completely intact (4/250 all-tied against the base's
0). R2's mixture inherits each format's gain roughly additively and inherits the joint half's
collapse along with it.

No arm meets the pre-registered criterion on T1-JOINT under the primary tie-aware convention; the
best is r5-joint at +0.121, p = 0.078. And the one arm that eliminates the E3 joint-vs-independent
gap does so by degrading the independent axis, not by repairing joint calibration. Taken with R2,
the batch's conclusion is that supervised exposure to human-anchored intensity labels at this scale
teaches this model to *predict the label distribution* — increasingly by emitting one value per
video — rather than to rank moments within a video.

If the goal for the workshop is the best-behaved fine-tuned arm rather than the largest headline
number, **r5-indep is the defensible choice**: it is the only arm with near-complete scoring
coverage (246/250 joint, so its mean is not a subset artefact), it does not regress any task, and
its improvement survives the honest convention. All results are single-seed (C9).

---

## Exact commands run

```bash
# 1. split R2's data by format (no rebuild)
.venv/bin/python -m finetune.split_by_format

# 2. trainings (8 × A6000, sequential; 0.65 h + 2.35 h)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/torchrun --nproc_per_node 8 -m finetune.train_lora --base Qwen/Qwen3.5-9B \
    --train finetune/data/train_indep.jsonl --val finetune/data/val_indep.jsonl \
    --out finetune/ckpt/r5-indep-5ep --epochs 5 --lr 1e-4 --rank 16 --seed 0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/torchrun --nproc_per_node 8 -m finetune.train_lora --base Qwen/Qwen3.5-9B \
    --train finetune/data/train_joint.jsonl --val finetune/data/val_joint.jsonl \
    --out finetune/ckpt/r5-joint-5ep --epochs 5 --lr 1e-4 --rank 16 --seed 0

# 3. eval — LoRA only, base reused. Two rounds of two concurrent runs (4 GPUs each):
#    round 1 = both arms' T1-JOINT (~33 min), round 2 = both arms' T1-INDEP (~7 min).
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m gold_eval.run_ranking             --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r5-indep-5ep/best"
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python -m gold_eval.run_ranking             --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r5-joint-5ep/best"
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m gold_eval.run_ranking_independent --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r5-indep-5ep/best"
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python -m gold_eval.run_ranking_independent --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r5-joint-5ep/best"

# 4. paired tests (tie-aware primary) + summary
.venv/bin/python -m finetune.paired_test --lora finetune/ckpt/r5-indep-5ep/best \
    --out gold_eval/results/finetune_paired_r5_indep.json
.venv/bin/python -m finetune.paired_test --lora finetune/ckpt/r5-joint-5ep/best \
    --out gold_eval/results/finetune_paired_r5_joint.json
.venv/bin/python -m gold_eval.summarize
```

Wall-clock: r5-indep training 0.65 h; r5-joint training 2.35 h; all four evals 40 min (two
concurrent rounds); paired tests + summarize < 1 min. **Total ≈ 3.7 h.**

---

## Deviations from the R5 spec

**Nothing under `gold_eval/tasks/` was touched; no eval prompt or frame protocol was altered.**

1. **The evaluated checkpoints are the early-stopped ones** (r5-indep step 200 of 305; r5-joint
   step 100 of 135), not the 5-epoch endpoints — `load_best_model_at_end` as configured in R2 and
   run 1. Same note as R2 Deviation #1: 5 epochs were optimised, an earlier checkpoint was selected.
2. **r5-joint received only two validation evaluations** (see Training). `eval_steps=100` was held
   at R2's value per "all else = R2" rather than tightened for the shorter run; flagged as a
   limitation on that arm's checkpoint selection rather than silently fixed.
3. **`split_by_format.py` was committed** rather than run as a throwaway snippet, so the ablation's
   inputs are reproducible.
4. Base T1 rows reused from run 1, as instructed. Single seed (0) for both arms.
5. `summarize.py` rebuilds `summary.json` from scratch; verified against the committed copy that all
   pre-existing rows are bit-identical and only the four new `r5-*` rows were added.
