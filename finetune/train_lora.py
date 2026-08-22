"""LoRA SFT for the intensity task on a Qwen VL base (finetune/PLAN.md §4).

Multi-image chat samples from build_sft_data.py; loss on assistant tokens only; LoRA on the
language model's projections, vision tower frozen. Single-node multi-GPU via torchrun (DDP).

    torchrun --nproc_per_node 8 -m finetune.train_lora \
        --base Qwen/Qwen3.5-9B --train finetune/data/train.jsonl --val finetune/data/val.jsonl \
        --out finetune/ckpt/qwen3.5-9b-lora-r16 --epochs 3 --lr 1e-4

Untested on the target node as of 2026-08-21 — the two model/processor loader lines are the
most likely place to need a class-name adjustment for the installed transformers version.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments,
                          EarlyStoppingCallback)


class SFTData(Dataset):
    def __init__(self, path: Path, processor, max_tokens: int):
        self.rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.proc, self.max_tokens = processor, max_tokens

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        images = [Image.open(p).convert("RGB") for p in r["images"]]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": r["prompt"]}]
        msgs_prompt = [{"role": "user", "content": content}]
        msgs_full = msgs_prompt + [{"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]}]
        prompt_text = self.proc.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)
        full_text = self.proc.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
        enc = self.proc(text=[full_text], images=images, return_tensors="pt")
        n_prompt = self.proc(text=[prompt_text], images=images, return_tensors="pt")["input_ids"].shape[1]
        ids = enc["input_ids"][0]
        if ids.shape[0] > self.max_tokens:
            return self[(i + 1) % len(self)]          # skip over-long samples (PLAN §8)
        labels = ids.clone(); labels[:n_prompt] = -100  # assistant tokens only
        out = {k: v[0] for k, v in enc.items()}
        out["labels"] = labels
        return out


def collate(batch):  # per-device batch size 1 — keep tensors as-is
    assert len(batch) == 1
    return {k: v.unsqueeze(0) if v.dim() in (1,) and k in ("input_ids", "attention_mask", "labels") else
            (v.unsqueeze(0) if k in ("input_ids", "attention_mask", "labels") else v) for k, v in batch[0].items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="Qwen/Qwen3.5-9B")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    proc = AutoProcessor.from_pretrained(a.base)
    model = AutoModelForImageTextToText.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    for n, prm in model.named_parameters():        # freeze everything; LoRA adds trainables
        prm.requires_grad = False
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                      # language-model projections only (regex excludes the vision tower)
                      target_modules=r".*(language_model|model\.layers).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    train_ds, val_ds = SFTData(a.train, proc, a.max_tokens), SFTData(a.val, proc, a.max_tokens)
    args = TrainingArguments(
        output_dir=str(a.out), per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, logging_steps=10,
        eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100,
        save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, report_to="none", seed=a.seed, remove_unused_columns=False,
        dataloader_num_workers=2, ddp_find_unused_parameters=False)
    tr = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                 data_collator=collate, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    tr.train()
    model.save_pretrained(str(a.out / "best"))
    proc.save_pretrained(str(a.out / "best"))
    (a.out / "train_args.json").write_text(json.dumps(vars(a), default=str, indent=2))
    print(f"[train] saved LoRA adapter -> {a.out/'best'}")


if __name__ == "__main__":
    main()
