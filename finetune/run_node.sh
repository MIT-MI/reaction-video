#!/usr/bin/env bash
# One-shot driver for the GPU node (8xA6000). Idempotent; rerun after any interruption.
#   bash finetune/run_node.sh /path/to/reaction_video_benchmark_phase0_pilot
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET_REPO="${1:?path to dataset repo (for reports/phase0/finetune)}"
BASE="${BASE:-Qwen/Qwen3.5-9B}"
OUT="finetune/ckpt/$(echo "$BASE" | tr '/' '_')-lora-r16"

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q "torch" "transformers>=4.50" "peft" "accelerate" "pillow" "scipy" "numpy" "tinker" "tinker-cookbook" 2>&1 | tail -1
command -v ffmpeg >/dev/null || { echo "ffmpeg missing"; exit 1; }

echo "=== 1. SFT data (downloads ~1.5k face videos from R2, cuts frames) ==="
.venv/bin/python -m finetune.build_sft_data --split "$DATASET_REPO/reports/phase0/finetune"

echo "=== 2. LoRA training ==="
.venv/bin/torchrun --nproc_per_node "${NGPU:-8}" -m finetune.train_lora \
  --base "$BASE" --train finetune/data/train.jsonl --val finetune/data/val.jsonl --out "$OUT" \
  --epochs "${EPOCHS:-3}" --lr "${LR:-1e-4}" --rank 16

echo "=== 3. Eval: base vs LoRA through the SAME local path (T1 joint + indep, 250 videos) ==="
for M in "hf:$BASE" "hf:$BASE+$OUT/best"; do
  .venv/bin/python -m gold_eval.run_ranking --model "$M"
  .venv/bin/python -m gold_eval.run_ranking_independent --model "$M"
done
.venv/bin/python -m gold_eval.summarize
echo "=== done; commit gold_eval/results and finetune/ckpt/*/train_args.json ==="
