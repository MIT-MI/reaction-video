# LoRA fine-tune run 4 / R4 — multi-task supervision (Qwen3.5-9B)

Executed 2026-08-25/26 against `finetune/PLAN.md` and the R4 batch spec. Every number below comes
from a file in this repo; nothing is quoted from memory or from a previous run.

**Question.** Does joint supervision across intensity + match beat single-task supervision?

**Answer: no.** R4 matches the single-task runs on every task and beats none of them. Against R2
(intensity-only) on T1 the difference is +0.022 / +0.020 (Wilcoxon p = .69 / .64); against R3
(match-only) on T2 it is −0.009 (McNemar p = 1.000). R4 *does* become the first run to clear the
pre-registered T1-JOINT bar (Δ +0.1335, p = .0301) where R2 did not (Δ +0.1116, p = .171) — but
the R4−R2 difference is itself p = .69, so which side of p < .05 a run lands on is where the noise
fell, not evidence that multi-task supervision helps.

---

## 1. Master table

Rows are LoRA arms, all against the same base and the same eval sets. T1 ρ is the primary
tie-aware convention (`per_video_rhos(tied_as_zero=True)`, n = 250 videos). T2 is accuracy on the
REBUILT 112-task match set.

| arm | trained on | T1-JOINT ρ | T1-INDEP ρ | T2 acc (rebuilt) |
|---|---|---|---|---|
| base `hf:Qwen/Qwen3.5-9B` | — | 0.0236 | 0.2534 | 0.3036 |
| **R2** order/intensity, 5 ep | indep + joint | 0.1352 | 0.3730 | 0.4018 |
| **R3** match-only, 3 ep | match | 0.0160 | 0.2237 | 0.4107 |
| **R4** multi-task, 5 ep | indep + joint + match | **0.1571** | **0.3931** | 0.4018 |

Paired against the base (Wilcoxon for T1, exact McNemar for T2):

| arm | T1-JOINT Δ (p) | T1-INDEP Δ (p) | T2 Δ (p) |
|---|---|---|---|
| R2 | +0.1116 (.171) | +0.1196 (.052) | +0.0982 (.071) |
| R3 | −0.0076 (.609) | −0.0298 (.247) | +0.1071 (.036) |
| R4 | **+0.1335 (.030)** | **+0.1397 (.008)** | +0.0982 (.090) |

Head-to-head, multi-task vs the matching single-task run — **this is R4's actual question**:

| comparison | task | single-task | R4 | Δ | p |
|---|---|---|---|---|---|
| R4 vs R2 | T1-JOINT | 0.1352 | 0.1571 | **+0.0219** | **0.695** |
| R4 vs R2 | T1-INDEP | 0.3730 | 0.3931 | **+0.0201** | **0.641** |
| R4 vs R3 | T2 | 0.4107 | 0.4018 | **−0.0089** | **1.000** |

Three nulls. Multi-task supervision is indistinguishable from single-task supervision on every
task it was trained for. It does not interfere either — the mixture costs nothing.

The R3 row (T1) and the R2 row (T2) are **off-diagonal fills run beyond the batch spec**; see §6.
They are what makes the table interpretable, and §4 is the reason.

## 2. Data and training

`train_multi.jsonl` — the same file built for R3, unmodified:

| split | indep | joint (×2) | match | total |
|---|---|---|---|---|
| train | 1,952 | 856 | 93 | **2,901** |
| val | 213 | 48 | 10 | **271** |

Disjointness was asserted in R3 and re-used unchanged: 0 overlap between the match rows and
`gold_eval/tasks/match_set.json` (candidate_id and video_id), and 0 overlap between the intensity
rows and the 250-video `ranking_set.json`. Nothing under `gold_eval/tasks/` is trained on.

`torchrun --nproc_per_node 8 -m finetune.train_lora --epochs 5 --lr 1e-4 --rank 16 --seed 0`.
Trainable params 29,097,984 (0.3083%), identical to R2/R3/R5.

| rows | steps | wall-clock | train loss | best eval loss |
|---|---|---|---|---|
| 2,901 | **455** | **5 h 23 m** | 1.000 → 0.175 | **0.4105 @ step 200** |

**The mixture overfits well before 5 epochs.** Eval loss by step:

| step | 100 | **200** | 300 | 400 | 455 |
|---|---|---|---|---|---|
| eval_loss | 0.4460 | **0.4105** | 0.4593 | 0.5586 | 0.5659 |

Minimum at step 200 (≈ 2.2 epochs), then monotonic increase to 0.5659 — a 38% degradation over
the back half of the run. `load_best_model_at_end` selected step 200, so **R4 is evaluated at
step 200**, and `best/` is byte-identical to `checkpoint-200` (md5 `2d9fe652…`, verified). The
spec's 5 epochs was ≈ 2.3× more than this mixture wants; the extra 3 h of compute produced only
worse checkpoints. A future run on this mixture should use 2–3 epochs.

## 3. T1: the gain is real, but ~22% of it is refusal, not ranking

The fine-tunes collapse to a **constant score across all moments** on a large fraction of videos.
Under the tie-aware convention an all-tied prediction is credited ρ = 0, so collapsing converts a
possibly-negative score into a free zero.

| arm | T1-JOINT all-tied | pred SD | distinct values |
|---|---|---|---|
| base | **0** / 250 | 27.89 | 28 |
| R2 | 136 / 250 | — | — |
| R3 (no intensity supervision) | **0** / 250 | — | — |
| R4 | **153** / 250 | 11.82 | 13 |

Decomposing R4's +0.1335 on T1-JOINT:

| component | n | base ρ | R4 ρ | contribution to Δ |
|---|---|---|---|---|
| videos R4 collapsed on (refusal → 0) | 153 | **−0.0490** | 0.0000 | **+0.0300** |
| videos R4 actually ranked | 97 | +0.1381 | **+0.4048** | **+0.1035** |

So ≈ 22% of the headline is the refusal effect and ≈ 78% is a genuine, large ranking improvement
(ρ 0.138 → 0.405) on the videos R4 does rank.

**The obvious objection — that the 97 are cherry-picked easy videos — does not hold, and in fact
runs the other way.** The base scores 0.1381 on that subset versus 0.0236 overall, so the subset
is *easier for the base*; the drop-ties Δ = +0.2668 (p = .0030, n = 97) is therefore measured
against a **stronger** baseline than the full set, not a weaker one. Selection deflates the
reported gain rather than inflating it.

T1-INDEP is cleaner: collapse contributes **−0.0146** there (the base is positive, +0.0537, on
the collapsed videos), so the entire +0.1397 and more comes from ranking (0.3280 → 0.5400 on the
182 ranked videos).

**The collapse is caused by intensity supervision specifically, not by LoRA training in general.**
R3 — trained only on match — collapses on **0/250** T1-JOINT videos, exactly like the base. Only
the arms trained on scalar intensity targets (R2: 136, R4: 153) regress toward a constant. This
is the classic SFT-on-scalars failure mode and it should be reported as a cost of the recipe.

## 4. T2: the gain is a format artefact — now with a direct experimental control

R3 concluded that the T2 gain was output-format compliance, not matching, on the strength of a
parsed-subset null plus a positional-prior argument. R4's batch adds the control that settles it.

| arm | match supervision | T2 acc (all 112) | unparsed | base-parsed subset (n = 95) | McNemar p |
|---|---|---|---|---|---|
| base | — | 0.3036 | **17** | 0.3579 | — |
| **R2** | **none — never saw a match example** | **0.4018** | **0** | 0.3684 (Δ +0.0105) | **1.0000** |
| R3 | 93 rows | 0.4107 | 0 | 0.3684 (Δ +0.0105) | **1.0000** |
| R4 | 93 rows | 0.4018 | 0 | **0.3579 (Δ ±0.0000)** | **1.0000** |

**R2 was trained purely on intensity ranking and never saw a single match example, yet it scores
identically to the match-trained arms on T2 (0.4018 vs 0.4018), fixes unparsed 17 → 0 identically,
and shows the identical null on the items the base actually answered.**

A "gain" that is fully reproduced by training on an unrelated task is not task skill. The entire
T2 effect across R2, R3 and R4 is the LoRA learning to emit a parseable letter instead of prose
(the base emits things like `"The viewer's reaction is a classic expression of disgust…"`). On
R4 the null is exact: 34/95 for both arms, discordant 12/12.

R4's headline also fails to clear p < .05 on its own (p = .0895) — unlike R3's .0357 — so it is
less likely to be misread out of context. Neither should be quoted as a matching result.

## 5. Verdict

- **Multi-task supervision does not beat single-task supervision.** R4 ≈ R2 on T1 (p = .69, .64)
  and R4 ≈ R3 on T2 (p = 1.000). It also does not hurt: no interference from mixing the formats.
- R4 clearing the pre-registered T1-JOINT bar where R2 missed it is **not** a multi-task effect —
  the two runs differ by p = .69. Reporting "multi-task crossed the threshold" would be a
  threshold artefact.
- The T1 intensity gain is **real** (≈ 78% of it), and larger than the headline suggests on the
  videos the model actually ranks (ρ 0.138 → 0.405), but it carries a **prediction-collapse cost**
  (153/250 videos constant-scored) that the tie-aware convention rewards rather than penalises.
- The T2 gain is **entirely a format artefact**, now proven by a training-data control (R2), not
  just by subgroup analysis.
- **Cross-task transfer is nil in both directions.** Match supervision does nothing for intensity
  (R3 on T1: −0.008, −0.030, both n.s.); intensity supervision does nothing for matching beyond
  format (R2 on T2 parsed subset: +0.011, p = 1.000).

## 6. Deviations and incidents

1. **R4 training exited non-zero but the artefact is intact and verified.** All 455 steps
   completed and rank 0 saved `best/`; ranks 1, 3, 4, 5, 6, 7 then hit `CUDA error: out of memory`
   inside `Trainer._load_best_model()` — strictly *after* training, while reloading the best
   checkpoint onto each GPU with the training allocator still resident. `torchrun` reported
   `exitcode 1`. Verified rather than assumed: `best/adapter_model.safetensors` is byte-identical
   (md5 `2d9fe652a9d840f5f8e61462fa94b4e5`) to `checkpoint-200`, which `trainer_state.json` names
   as `best_model_checkpoint`. Nothing is partial or corrupt.
2. **≈ 8 h of idle GPU time lost.** The crash in (1) killed the run at 03:10 and no waiter was
   armed on it, so the node sat idle until 11:00. Monitors on the eval runs were subsequently
   armed with failure signatures (`Traceback|CUDA error|out of memory|Killed`), not just success
   markers, so a crash surfaces instead of reading as silence.
3. **Effective epochs ≠ spec epochs.** Spec said 5 epochs; the mixture's best checkpoint is at
   step 200 ≈ 2.2 epochs (§2). Trained as specified, evaluated at the best checkpoint as the
   training config dictates. Flagged because "R4 = 5 epochs" is misleading.
4. **Off-diagonal fills run beyond spec.** The spec's master table left two cells unrunnable (R2
   was never evaluated on T2; R3 never on T1). Three extra evals were added — R2→T2, R3→T1-JOINT,
   R3→T1-INDEP — costing ≈ 45 min of otherwise-idle GPU. They changed no required number, and
   they are what produced §4's control and §3's finding that collapse is intensity-specific.
5. **T2 accuracies come from `paired_test --match` reading the raw jsonl + task file directly,
   not from `results/match/scores.json`.** `score_match` rewrites that file with only the arms it
   scored (flagged in REPORT_R3 §7.4), so it is not a reliable source mid-batch.
6. **The R3→T1-INDEP fill shared GPUs with unrelated `grid160` jobs** started on the node at
   11:47/12:09. This slowed that eval (3.7 s/item vs 1.6) but cannot affect its outputs —
   decoding is greedy and the frame cache was fully warm (no cache writes during any eval, so the
   `frames.py` race fixed in `f9d513f` was never exercised).

## 7. Wall-clock

| step | wall-clock |
|---|---|
| R4 training (455 steps, 8× A6000) | **5 h 23 m** |
| R4 T1-JOINT + T1-INDEP (concurrent, 4 GPUs each) | **31 m 28 s** |
| R4 T2-MATCH (4 GPUs, clips pre-warmed) | **4 m 31 s** |
| fill: R2→T2 | ≈ 4 m |
| fill: R3→T1-JOINT | ≈ 34 m |
| fill: R3→T1-INDEP (GPU-contended) | ≈ 21 m |
| paired tests + analysis (CPU) | ≈ 3 m |

## 8. Reproduction

```bash
.venv/bin/torchrun --nproc_per_node 8 -m finetune.train_lora \
  --base Qwen/Qwen3.5-9B \
  --train finetune/data/train_multi.jsonl --val finetune/data/val_multi.jsonl \
  --out finetune/ckpt/r4-multi-5ep --epochs 5 --lr 1e-4 --rank 16 --seed 0

CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m gold_eval.run_ranking             --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r4-multi-5ep/best"
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python -m gold_eval.run_ranking_independent --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r4-multi-5ep/best"
.venv/bin/python -m gold_eval.run_match --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r4-multi-5ep/best"

.venv/bin/python -m finetune.paired_test         --lora finetune/ckpt/r4-multi-5ep/best --out gold_eval/results/finetune_paired_r4.json
.venv/bin/python -m finetune.paired_test --match --lora finetune/ckpt/r4-multi-5ep/best --out gold_eval/results/finetune_paired_r4_match.json

# off-diagonal fills (§6.4)
.venv/bin/python -m gold_eval.run_match           --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r2-order-5ep/best"
.venv/bin/python -m gold_eval.run_ranking         --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r3-match-3ep/best"
.venv/bin/python -m gold_eval.run_ranking_independent --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r3-match-3ep/best"
.venv/bin/python -m finetune.paired_test         --lora finetune/ckpt/r3-match-3ep/best  --out gold_eval/results/finetune_paired_r3.json
.venv/bin/python -m finetune.paired_test --match --lora finetune/ckpt/r2-order-5ep/best  --out gold_eval/results/finetune_paired_r2_match.json
```
