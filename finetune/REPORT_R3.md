# LoRA fine-tune run 2 / R3 — match-only, weak auto-aligned supervision (Qwen3.5-9B)

Executed 2026-08-25 against `finetune/PLAN.md` and the R3 batch spec. Every number below comes
from a file in this repo; nothing is quoted from memory or from a previous run.

**Question.** Can weak auto-aligned supervision (temporal alignment, no human labels) teach
same-video trigger matching?

**Answer: no.** The headline +0.107 accuracy gain is significant on its face (McNemar p = .036)
but is **entirely a format artefact**. Restricted to the items the base model actually answered,
the effect is exactly null (Δ = +0.011, McNemar b=8 c=9, **p = 1.000**). What the fine-tune
reliably learned is to emit a parseable letter (17 unparsed → 0) and to copy the training
positional prior — not to match a reaction to its trigger.

---

## 1. Data

`build_sft_data --match_train` over the office build
`reports/phase0/finetune/match_train.json` (**103 tasks**), F-MATCH format mirroring
`run_match.build_parts` exactly: `[prompt, REACTION frames ×12, SEGMENT A–D frames ×6 each,
answer cue] → single letter`. Labels are auto temporal alignment — weak supervision, disclosed.

| split | indep | joint (×2) | **match** | total |
|---|---|---|---|---|
| train | 1,952 | 856 | **93** | 2,901 |
| val | 213 | 48 | **10** | 271 |

Split is by `sha1(candidate_id) % 10 == 0` → val. 0 failed items. R3 trains on the `match`
rows only (`train_match.jsonl` / `val_match.jsonl`); the full files are kept as
`train_multi.jsonl` / `val_multi.jsonl` for R4.

**Disjointness assert (required by the spec, passed):**

| | candidates | videos |
|---|---|---|
| eval `match_set.json` (rebuilt) | 112 | 97 |
| R3 train+val match rows | 103 | 92 |
| **overlap** | **0** | **0** |

Asserted on both `candidate_id` and `video_id`, not assumed from the construction. A separate
check confirmed the R4 mixture's intensity rows also have **0** overlap with the 250-video T1
`ranking_set.json`.

## 2. Training

`torchrun --nproc_per_node 8 -m finetune.train_lora --base Qwen/Qwen3.5-9B --epochs 3 --lr 1e-4
--rank 16 --seed 0` — hparams as R2 except epochs (3, per spec). Trainable params 29,097,984
(0.3083%), identical to R2/R5.

| rows | steps | wall-clock | final eval loss |
|---|---|---|---|
| 93 | **9** (3/epoch) | **430 s** | 0.4380 |

Only 9 optimiser steps — 93 rows at an effective batch of 32. This is a very small amount of
supervision and the result below should be read with that in mind: R3 tests *this* weak-label
pool, not the format's ceiling.

## 3. T2 accuracy on the REBUILT match set

Both arms run fresh on the rebuilt 112-task set (`fetch_tasks` → 112; the option re-cut means
pre-rebuild base rows are not comparable). Sequential, `HFLocalBackbone` greedy.

| arm | n | correct | accuracy | 95% Wilson | unparsed |
|---|---|---|---|---|---|
| base `hf:Qwen/Qwen3.5-9B` | 112 | 34 | 0.3036 | [0.2261, 0.3941] | **17** |
| **R3 LoRA** (match-only, 3 ep) | 112 | 46 | **0.4107** | [0.3240, 0.5033] | **0** |

Chance = 0.25.

**Exact McNemar on the 112 shared items** (`paired_test.py --match`, same statistic as
`summarize.py → match_paired`):

| both correct | base only | LoRA only | both wrong | discordant | Δ acc | p (exact) |
|---|---|---|---|---|---|---|
| 26 | **8** | **20** | 58 | 28 | **+0.1071** | **0.0357** |

Taken alone this clears a conventional bar. It should not be reported that way — §4 is the
result.

## 4. The gain is a format artefact, not matching skill

The base leaves **17/112 items unparsed**; the LoRA leaves 0. Unparsed rows have
`pred_index = None`, which never equals `correct_index`, so they are scored as misses for the
base — correctly, but that means 17 items are decided by output format rather than by matching.
The base's unparsed outputs are prose, not wrong letters:

> `"The viewer's reaction is a classic expression of disgust, often referr…"`
> `"The viewer's reaction frames show a sequence of intense laughter, culm…"`

**Restricting to the 95 items the base actually answered removes the entire effect:**

| subset | n | base acc | R3 LoRA acc | Δ | McNemar b/c | p (exact) |
|---|---|---|---|---|---|---|
| all shared items | 112 | 0.3036 | 0.4107 | +0.1071 | 8 / 20 | **0.0357** |
| **base parsed only** | **95** | **0.3579** | **0.3684** | **+0.0105** | **8 / 9** | **1.0000** |

Δ collapses from +0.107 to +0.011 and the discordant split goes from 8/20 to 8/9 — as close to a
perfect null as this test produces. **The fine-tune did not learn to match; it learned to answer
in the required format.**

### The residual is consistent with a positional prior

The remaining signal lives in the 17 base-unparsed items (LoRA 11/17 = 0.647), but that subgroup
is enriched for the answer the LoRA over-predicts and is far too small to carry a claim:

| | gold index 0 | 1 | 2 | 3 | "always answer 3" |
|---|---|---|---|---|---|
| TRAIN labels (n=103) | 20 | 25 | 28 | **30** | — |
| EVAL labels (n=112) | 32 | 21 | 23 | **36** | 0.3214 |
| R3 LoRA predictions | 27 | 26 | 22 | **37** | — |
| base predictions (parsed) | 22 | 22 | **35** | 16 | — |

Both the training pool and the rebuilt eval set are skewed toward index 3, and the LoRA's
prediction distribution tracks the label marginal closely while the base's does not. On the 17
base-unparsed items specifically, gold is 7/17 index-3, so **"always answer 3" alone scores
0.412** there — inside the LoRA's 95% Wilson interval [0.413, 0.827] for that subgroup. Nothing
in that 11/17 is distinguishable from the prior.

Against the relevant baselines the LoRA's overall 0.4107 is much less impressive than it looks:

| baseline | accuracy |
|---|---|
| uniform chance | 0.2500 |
| sample from the eval label marginal | 0.2623 |
| **always answer "D" (index 3)** | **0.3214** |
| base (parsed subset) | 0.3579 |
| **R3 LoRA** | **0.4107** |

## 5. Context: where both arms sit among the T2 arms

From `results/match/scores.json`, all on the same rebuilt 112-task set:

| arm | accuracy | unparsed |
|---|---|---|
| gemini-3.7-flash `__video` | 0.5446 | 0 |
| gpt-5 | 0.5089 | 0 |
| gemini-3.7-flash | 0.5000 | 1 |
| Qwen3.6-35B-A3B | 0.4821 | 0 |
| Qwen3.6-27B | 0.4464 | 0 |
| **R3 LoRA (this run)** | **0.4107** | 0 |
| Qwen3.5-397B-A17B | 0.4107 | 0 |
| Kimi-K2.6 | 0.3750 | 0 |
| **base hf Qwen3.5-9B (this run)** | **0.3036** | 17 |
| Qwen3.5-9B via tinker | 0.2589 | 20 |

The two independent runs of the same 9B base (hf 0.3036/17 unparsed, tinker 0.2589/20 unparsed)
agree that this model's T2 weakness is substantially a format-compliance problem — which is
exactly what R3's LoRA fixes, and exactly why the fix must not be read as a matching gain.

Error temporal distance is unchanged by the fine-tune (base mean 27.95 s / median 26.67 s; R3
mean 29.50 s / median 27.73 s) — the LoRA's errors are not landing closer to the correct segment,
which is what a genuine matching improvement would predict.

## 6. Verdict

- **R3 does not show that weak auto-aligned supervision teaches trigger matching.** On the
  shared parseable subset the effect is null (Δ +0.011, p = 1.000).
- The reportable, robust effect is **format compliance**: 17 unparsed → 0.
- The headline +0.107 / p = .036 must **not** be quoted as a matching result. If it appears in
  the paper it has to carry the parsed-subset null next to it, the same way R2's single-seed
  +0.112 carries its cross-seed null.
- 9 optimiser steps on 93 weak-label rows is a weak test of the format. This is evidence about
  *this* supervision pool, not proof that F-MATCH cannot work with more/better labels.

## 7. Deviations and incidents

1. **Prerequisite 3 abort condition not taken, deliberately.** The spec says to abort if
   `git diff gold_eval/tasks/match_set.json` is empty after `fetch_tasks`. It *was* empty, but
   the premise was inverted: the office committed the rebuilt set in `1d466af` before this
   session pulled, so Supabase and the working tree already agreed. Verified positively instead
   — 150 → 112 tasks and **all 99 shared candidates have re-cut options** — then proceeded.
2. **Frame-cache concurrency bug found and fixed** (`gold_eval/frames.py`, committed in
   `f9d513f`). `fetch_clip` and `extract_frames` both wrote a fixed `<name>.tmp` and renamed it
   into place, so two eval arms sharing the cache raced. The retry wrapper absorbed it as a 15 s
   stall, but **a task exhausting its 4 retries would have silently desynced the two arms' item
   sets.** Temps are now PID-unique. Both arms were nonetheless re-run **sequentially** for this
   report, so no concurrent-cache path contributed to the numbers above.
3. **GPU node wedged mid-run; node rebooted.** The first eval attempt died when the NVIDIA
   driver deadlocked (a trainer rank stuck in `exit_mm` held the address-space rwsem; all
   subsequent CUDA inits and `nvidia-smi` calls blocked unkillably in D state). Admin rebooted.
   No partial rows survived and nothing in this report comes from the aborted attempt — both
   arms were run start-to-finish after the reboot. The 560 eval clips were pre-warmed during the
   outage (CPU/network only), so both arms were GPU-bound: **517 s for both**.
4. **`score_match` truncates `scores.json`.** It rewrites the file with only the arms it scored,
   so the office's earlier `--pred_file` XV run had already reduced the committed file to a
   single `__match_xv` key, and a plain re-run drops that key. The committed file here is the
   **union** — 10 re-scored arms plus the office's `tinker_Qwen_Qwen3.6-35B-A3B__match_xv` entry
   preserved. Flagging rather than changing shared tooling mid-batch; the office may want
   `score_match` to merge instead of overwrite.

## 8. Reproduction

```bash
.venv/bin/python -m finetune.build_sft_data \
  --split ../reaction_video_benchmark_phase0_pilot/reports/phase0/finetune \
  --joint_cap 24 \
  --match_train ../reaction_video_benchmark_phase0_pilot/reports/phase0/finetune/match_train.json
# -> finetune/data/{train,val}.jsonl; copied to *_multi.jsonl, format=="match" filtered to *_match.jsonl

.venv/bin/torchrun --nproc_per_node 8 -m finetune.train_lora \
  --base Qwen/Qwen3.5-9B \
  --train finetune/data/train_match.jsonl --val finetune/data/val_match.jsonl \
  --out finetune/ckpt/r3-match-3ep --epochs 3 --lr 1e-4 --rank 16 --seed 0

.venv/bin/python -m gold_eval.run_match --model "hf:Qwen/Qwen3.5-9B"
.venv/bin/python -m gold_eval.run_match --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r3-match-3ep/best"

.venv/bin/python -m finetune.paired_test --match \
  --lora finetune/ckpt/r3-match-3ep/best \
  --out gold_eval/results/finetune_paired_r3_match.json
```

Wall-clock: data build ~20 min (cold clip/frame cache); training 430 s; both T2 arms 517 s
(clips pre-warmed).
