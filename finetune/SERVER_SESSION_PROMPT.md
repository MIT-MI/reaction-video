# Server session — paste-ready brief for the Claude session on the GPU node

Expected wall-clock (8×A6000): env 10 min · data build 45–75 min (≈1.5k face videos from R2
+ ffmpeg frames) · LoRA training 1–3 h (2.8k samples/epoch × 3 epochs; ~2 s/sample/GPU) ·
eval base+LoRA on T1 joint+indep ≈ 1.5 h · total **≈ 4–6 h**, one afternoon. Data build and
training are resumable; rerun the same command after any interruption.

Prerequisites on the node: git, python3.10+, ffmpeg, CUDA torch; HF access to
`Qwen/Qwen3.5-9B` (`huggingface-cli login` if gated). Copy `TINKER_API_KEY1` only if you
want the Tinker fallback.

---------------------------------------------------------------- PASTE BELOW THIS LINE

You are running the fine-tuning experiment for ReactionBench (NeurIPS 2026 IAEval workshop,
deadline 2026-08-29). Read `finetune/PLAN.md` in full first — it is the task definition and
the pre-registered evaluation; do not change its design decisions. Then execute the steps
below end-to-end, verify each, and write the report. Work on branch `gold-eval`; commit
after each stage with the `<scope>: <summary>` convention; never commit `finetune/data/`,
`finetune/ckpt/` weights or `gold_eval/data/` (gitignored) — DO commit
`gold_eval/results/**` and `finetune/ckpt/*/train_args.json`.

Repos (clone side by side):
  git clone -b gold-eval https://github.com/MIT-MI/reaction-video.git
  git clone -b phase0/pilot https://github.com/quao627/reaction_video_benchmark.git reaction_video_benchmark_phase0_pilot
  cd reaction-video

Task definition (from PLAN.md §3): supervised LoRA on within-video reaction intensity.
  F-INDEP: ≤12 frames of a moment window + the exact E3 prompt → integer 0–100
  F-JOINT: ≤32 full-video frames + the exact T1 joint prompt → JSON {"m0":int,...}
  Labels = rater-debiased Stage-A consensus mapped to 0–100. Train 1,364 videos / 1,952
  moments, val 155 / 213, zero overlap with any eval set. Prompts are imported from the eval
  runners — never retype them.

Steps:
 1. `bash finetune/run_node.sh ../reaction_video_benchmark_phase0_pilot` runs everything
    (venv → build_sft_data → torchrun train_lora → eval base & LoRA → summarize). Prefer
    running the stages by hand the first time so you can inspect each:
    a. `.venv/bin/python -m finetune.build_sft_data --split ../reaction_video_benchmark_phase0_pilot/reports/phase0/finetune --limit 3`
       then without --limit. Check: train.jsonl has ≈1,952 indep + ≈856 joint rows, val ≈213 + joint;
       spot-check 3 rows (image paths exist, answers are ints / JSON). Record failed_videos.
    b. `torchrun --nproc_per_node 8 -m finetune.train_lora --base Qwen/Qwen3.5-9B --train finetune/data/train.jsonl --val finetune/data/val.jsonl --out finetune/ckpt/Qwen_Qwen3.5-9B-lora-r16 --epochs 3 --lr 1e-4 --rank 16`
       The loader uses AutoProcessor + AutoModelForImageTextToText; if the installed
       transformers needs a different class for Qwen3.5, change ONLY those two lines and note
       it in the report. Confirm trainable params ≈ 40–60M and that loss is masked to the
       assistant span (print one sample's labels != -100 count ≈ answer token count).
       Log train/val loss per 100 steps; keep the best-val checkpoint (`best/`).
    c. Eval — BOTH base and LoRA through the same local path (this is the control):
       `.venv/bin/python -m gold_eval.run_ranking --model "hf:Qwen/Qwen3.5-9B"`
       `.venv/bin/python -m gold_eval.run_ranking_independent --model "hf:Qwen/Qwen3.5-9B"`
       `.venv/bin/python -m gold_eval.run_ranking --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/Qwen_Qwen3.5-9B-lora-r16/best"`
       `.venv/bin/python -m gold_eval.run_ranking_independent --model "hf:Qwen/Qwen3.5-9B+finetune/ckpt/Qwen_Qwen3.5-9B-lora-r16/best"`
       Then `.venv/bin/python -m gold_eval.summarize`. Do NOT edit anything under gold_eval/tasks/.
    d. Paired test base vs LoRA on T1-JOINT and T1-INDEP: per-video Spearman from
       `gold_eval.score_ranking.per_video_rhos` on the intersection of scored videos;
       Wilcoxon signed-rank + seeded bootstrap CI of the mean Δ (copy the pattern in
       `gold_eval/summarize.py → ranking_paired`). Success criterion is pre-registered:
       Δρ_JOINT ≥ 0.10 with p < 0.05. A null result is reported as a null result.
 2. If time remains (same day): ablations INDEP-only and JOINT-only (`--train` on filtered
    jsonl), and seeds 1 and 2 of the main run; report mean±std if ≥2 seeds.
 3. Write `finetune/REPORT.md` with exactly these sections:
    - Setup: node, GPUs, transformers/peft versions, base model revision, any loader edits
    - Data: counts from build_sft_data, failed videos, sample token-length stats
    - Training: steps, epochs, best val loss & step, wall-clock, trainable params
    - Results table: rows {base (local), LoRA} × columns {T1-JOINT mean ρ [CI], T1-INDEP mean ρ [CI],
      n scored}; paired Δ with CI and p; human ceiling 0.649 as reference; the Tinker
      zero-shot Qwen3.5-9B rows (joint .029 / indep .334) quoted from gold_eval/results/summary.json
    - Honest interpretation in ≤5 sentences, including whether the pre-registered criterion was met
    - Exact commands run, and anything that deviated from PLAN.md
 4. Commit & push: `finetune: LoRA run 1 — <one-line result>` with REPORT.md, results/**,
    train_args.json. Then print the REPORT.md results table in your final message.

Rules: keep the evaluation prompts and frame protocol untouched; if a run must deviate
(OOM → lower --joint_cap to 24; class-name fixes), say so explicitly in REPORT.md; never
fabricate or back-fill numbers — every number must come from a file in results/.

---------------------------------------------------------------- RUN 2 MATRIX (GPU 8/24 AM; match source 8/24 PM)

Owner plan 2026-08-22. Two machines, two briefs. Deadline 8/29–30 → numbers frozen 8/27.

### Office machine (has pool.csv + source videos) — 8/24 afternoon, ~2–3 h, dataset repo
```
git pull                                   # branch phase0/pilot
# A. rebuild the frozen EVAL match set with the audited builder (pairwise >=10 s)
python tools/data_annotation/gold_platform/build_match_tasks.py --dry_run
python tools/data_annotation/gold_platform/build_match_tasks.py --force      # re-cuts + re-uploads match/ and upserts gold_match_tasks
# B. build the match TRAINING pool (non-eval videos) and its clips under match_train/
python scripts/pilot/build_match_train_pool.py --eval_dir ../reaction-video/gold_eval/tasks
python tools/data_annotation/gold_platform/build_match_tasks.py \
    --set_csv reports/phase0/stimulus_ranking/match_train_pool.csv \
    --train_out reports/phase0/finetune/match_train.json
git add reports/phase0/finetune/match_train.json reports/phase0/stimulus_ranking/match_train_pool.csv && git commit -m "match: rebuilt eval set + training items" && git push
```
Then on the laptop: `python -m gold_eval.fetch_tasks` (new eval match set) → `bash gold_eval/run_all.sh` match-only re-run for all 7 arms (~4 h) → G9 human pass.

### GPU node — run order (each step idempotent; commit REPORT_<run>.md + results after each)
| # | when | run | data | train | eval | ~time |
|---|---|---|---|---|---|---|
| R2 | 8/24 AM | intensity, order-aligned | `build_sft_data` (no --match_train) | INDEP+JOINT mix, 5 ep, seed 0 | T1 joint+indep, base vs LoRA, tie-aware paired | 6–7 h |
| R3 | 8/25 AM | match-only | `build_sft_data --match_train <match_train.json>` then filter `format==match` | F-MATCH, 3 ep | T2 (REBUILT set) base vs LoRA via hf: path; McNemar | 2–3 h |
| R4 | 8/25 PM | multi-task | INDEP+JOINT+MATCH mix | 5 ep | T1 + T2, base vs LoRA | 6–7 h |
| R5 | 8/26 | ablations | INDEP-only / JOINT-only | 5 ep each | T1 | 2×3 h |
| R6 | if time | seeds 1–2 of R2 | — | — | mean±std | 2×6 h |

Hard rules: (1) the eval sets in `gold_eval/tasks/` are never training data — F-MATCH rows
come ONLY from `match_train.json` (videos disjoint from every eval set by construction);
(2) after the office rebuild, run `python -m gold_eval.fetch_tasks` on the node BEFORE R3/R4
so T2 eval uses the rebuilt set; (3) base rows for the rebuilt T2 must be re-run through the
hf: path too; (4) tie-aware paired test primary, pre-registered drop-ties alongside; (5) one
REPORT per run, same template as REPORT.md; (6) never train on anything under gold_eval/tasks.

Not run, by design: side-by-side (SBS) training — frame-only models cannot consume the
hstacked video and the SBS arm is Gemini-only supplementary; rationale fine-tuning — no human
references outside the eval set (a distillation variant is possible later, disclosed as weak).
