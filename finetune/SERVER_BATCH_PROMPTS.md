# Server batch prompts — fine-tune runs, two batches (2026-08-23)

Batch 1 needs nothing new (runs now). Batch 2 needs `match_train.json` + the rebuilt eval
match set from the office machine (8/24 PM). Paste each batch into its own Claude session on
the GPU node. Commit after every completed run; **never push** (owner reviews and pushes).

================================================================ BATCH 1 — paste below

You are running fine-tune batch 1 for ReactionBench (workshop DDL 2026-08-29). This node
already ran run 1 (see finetune/REPORT.md); reuse its venv and caches. First:
`cd reaction-video && git checkout gold-eval && git pull`, and
`cd ../reaction_video_benchmark_phase0_pilot && git pull`. Read `finetune/PLAN.md` and
`finetune/REPORT.md` before touching anything.

Standing rules: never modify or train on anything under `gold_eval/tasks/`; eval prompts and
the frame protocol stay untouched; every reported number must come from a file under
`gold_eval/results/` or a ckpt's `train_args.json`; after each completed run `git add` the
run's REPORT + `gold_eval/results` + `finetune/ckpt/*/train_args.json` and commit as
`finetune: <run-id> — <one-line result>`, but DO NOT `git push`. Use
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for every torchrun. If anything deviates
from the spec (OOM workaround, API change), record it in that run's REPORT under Deviations.

## R2 — intensity, order-aligned, 5 epochs (the main rerun)
Task: same as run 1 (PLAN §3-§5: F-INDEP + F-JOINT×2 mixture, labels = debiased consensus on
0-100, LoRA r16 on Qwen/Qwen3.5-9B) with exactly two changes: (a) training samples now use
the EVAL part order — text prompt, then frames, then the short cue ("Score (0-100):" /
"JSON scores:") — the builder already emits this (`prompt_before` field); (b) 5 epochs
(run 1's val loss was still falling at its final step).
1. Rebuild data:
   `.venv/bin/python -m finetune.build_sft_data --split ../reaction_video_benchmark_phase0_pilot/reports/phase0/finetune --joint_cap 24`
   Verify: train ≈1,952 indep + 856 joint, val ≈213 + 48; a sampled row has `prompt_before`,
   integer/JSON answer, existing image paths; 0 failed videos (report if not).
2. Train:
   `torchrun --nproc_per_node 8 -m finetune.train_lora --base Qwen/Qwen3.5-9B --train finetune/data/train.jsonl --val finetune/data/val.jsonl --out finetune/ckpt/r2-order-5ep --epochs 5 --lr 1e-4 --rank 16 --seed 0`
   Verify trainable ≈29.1M, loss masked to the assistant span (print one row's unmasked decode).
3. Eval the LoRA (base T1 rows exist from run 1 — reuse, do not re-run):
   `.venv/bin/python -m gold_eval.run_ranking --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r2-order-5ep/best"`
   `.venv/bin/python -m gold_eval.run_ranking_independent --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r2-order-5ep/best"`
4. Extend `finetune/paired_test.py` so the TIE-AWARE convention is primary
   (`per_video_rhos(..., tied_as_zero=True)` over all 250 videos, crediting ρ=0 to all-tied
   predictions) with the pre-registered drop-ties result alongside, then run it vs
   `hf:Qwen/Qwen3.5-9B`. Also report prediction SD and all-tied counts (run 1's collapse
   diagnostic, REPORT.md "Output-range collapse" section is the template).
5. `.venv/bin/python -m gold_eval.summarize` → write `finetune/REPORT_R2.md` (same template
   as REPORT.md; quote run 1's numbers for comparison) → commit (no push).

## R5 — single-format ablations, 5 epochs each
Task: which format drives whatever R2 shows — INDEP-only vs JOINT-only, all else = R2.
1. Filter the R2 data files by the `format` field into
   `finetune/data/train_indep.jsonl` / `train_joint.jsonl` (+ matching val files) with a
   short python script (json per line; keep joint oversampling as-is).
2. Two trainings → `finetune/ckpt/r5-indep-5ep`, `finetune/ckpt/r5-joint-5ep` (seed 0).
3. Eval each on T1 joint + indep (LoRA only); tie-aware paired vs base for both.
4. `finetune/REPORT_R5.md` with a 4-row table (base, R2, r5-indep, r5-joint); commit (no push).

## R6 — seeds 1 and 2 of R2 (only if batch 2's data has NOT yet arrived)
If `../reaction_video_benchmark_phase0_pilot/reports/phase0/finetune/match_train.json`
exists, STOP here and leave the GPUs for batch 2; otherwise train R2 twice more with
`--seed 1` / `--seed 2` (`ckpt/r2-seed1`, `ckpt/r2-seed2`), eval T1 both, and append a
mean±std across seeds to REPORT_R2.md; commit (no push).

Final message: one table — every run in this batch × {T1-JOINT tie-aware ρ, Δ vs base [CI],
p; T1-INDEP same}, plus wall-clock per run.

================================================================ BATCH 2 — paste below
(only after the office machine has pushed match_train.json, ~8/24 PM)

You are running fine-tune batch 2 (match task) for ReactionBench. Same standing rules as
batch 1 (top of this file): no pushes, commit per run, nothing under gold_eval/tasks/ is
ever trained on, numbers only from files.

Prerequisites — verify ALL before training; abort with a clear message if any fails:
1. `cd ../reaction_video_benchmark_phase0_pilot && git pull` →
   `reports/phase0/finetune/match_train.json` exists (office build). Note its task count.
2. `cd ../reaction-video && git pull` (code current; batch-1 commits are local, so a plain
   pull may rebase — keep local commits intact).
3. Refresh the REBUILT eval match set from Supabase:
   `.venv/bin/python -m gold_eval.fetch_tasks` → match_set.json count printed; confirm the
   file changed (the rebuild re-cut the options; if `git diff --stat gold_eval/tasks/match_set.json`
   is empty, the office rebuild has not landed — abort).
4. Batch-1 trainings finished (GPUs free).

## R3 — match-only fine-tune
Task: can weak auto-aligned supervision teach same-video trigger matching? Train data =
`match_train.json` items ONLY (videos disjoint from every eval set by construction — assert
anyway). Format F-MATCH mirrors run_match exactly: interleaved [prompt, REACTION frames,
SEGMENT A-D frames, answer cue] → single letter.
1. Data: `.venv/bin/python -m finetune.build_sft_data --split ../reaction_video_benchmark_phase0_pilot/reports/phase0/finetune --joint_cap 24 --match_train ../reaction_video_benchmark_phase0_pilot/reports/phase0/finetune/match_train.json`
   Copy the produced files to `train_multi.jsonl` / `val_multi.jsonl` (R4 uses them), then
   filter `format=="match"` into `train_match.jsonl` / `val_match.jsonl`.
   ASSERT: zero overlap between train_match candidate/video ids and
   `gold_eval/tasks/match_set.json`; report both counts.
2. Train 3 epochs → `finetune/ckpt/r3-match-3ep` (seed 0; other hparams as R2).
3. Eval T2 on the REBUILT set — base AND LoRA fresh (the options changed, old base rows are
   for the old set):
   `.venv/bin/python -m gold_eval.run_match --model "hf:Qwen/Qwen3.5-9B"`
   `.venv/bin/python -m gold_eval.run_match --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/r3-match-3ep/best"`
4. Add a `--match` mode to `finetune/paired_test.py` (exact McNemar on shared items,
   following `summarize.py → match_paired`); run base vs LoRA.
5. `finetune/REPORT_R3.md` (accuracy + Wilson CI for both arms, McNemar b/c/p, unparsed
   counts); commit (no push).

## R4 — multi-task fine-tune
Task: does joint supervision across intensity + match beat single-task? Train on
`train_multi.jsonl` (INDEP + JOINT×2 + MATCH), 5 epochs → `finetune/ckpt/r4-multi-5ep`.
1. Train (seed 0).
2. Eval T1 joint + indep AND T2 (rebuilt set), LoRA only (bases exist from R2 reuse + R3).
3. Paired tests: tie-aware T1 vs base; McNemar T2 vs base.
4. `finetune/REPORT_R4.md` with the master table: rows {base, R2, R3, R4} × columns
   {T1-JOINT tie-aware ρ, T1-INDEP tie-aware ρ, T2 acc (rebuilt)}; commit (no push).

Final message: that master table + per-run wall-clock + any deviations.
