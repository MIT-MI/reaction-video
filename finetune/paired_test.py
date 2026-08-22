"""Paired base-vs-LoRA test on T1-JOINT and T1-INDEP (PLAN §5 success criterion).

Same statistics as `gold_eval.summarize.ranking_paired`, but pairing the two ARMS of the
fine-tune (local base vs LoRA) on one task instead of the two tasks of one model: per-video
Spearman from `score_ranking.per_video_rhos`, intersected over videos both arms scored, then
Wilcoxon signed-rank on the differences and a seeded percentile bootstrap CI of the mean delta
(RNG_SEED / N_BOOT imported from summarize so the numbers match the rest of the report).

Pre-registered criterion: delta rho on T1-JOINT >= 0.10 with Wilcoxon p < .05.

    python -m finetune.paired_test --lora finetune/ckpt/Qwen_Qwen3.5-9B-lora-r16/best
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from gold_eval.costs import slug
from gold_eval.score_ranking import per_video_rhos
from gold_eval.summarize import boot_mean_ci

HERE = Path(__file__).parent
RES = HERE.parent / "gold_eval" / "results"


def preds_for(sub: str, model: str) -> dict[str, dict] | None:
    f = RES / sub / f"{slug(model)}.jsonl"
    if not f.exists():
        return None
    return {r["video_id"]: r for r in
            (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def arm(items: list[dict], sub: str, model: str, key: str) -> dict[str, float] | None:
    p = preds_for(sub, model)
    return None if p is None else per_video_rhos(items, p, key=key)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="hf:Qwen/Qwen3.5-9B")
    p.add_argument("--lora", required=True, help="adapter dir, appended to the base with '+'")
    p.add_argument("--tasks", type=Path, default=RES.parent / "tasks/ranking_set.json")
    p.add_argument("--key", default="mean", choices=["mean", "debiased"],
                   help="consensus the rhos are taken against (summarize reports both)")
    p.add_argument("--out", type=Path, default=RES / "finetune_paired.json")
    a = p.parse_args()

    items = json.loads(a.tasks.read_text())
    ft = f"{a.base}+{a.lora}"
    report: dict = {"base": a.base, "lora": ft, "consensus_key": a.key, "tasks": {}}

    for label, sub in (("T1-JOINT", "ranking"), ("T1-INDEP", "ranking_indep")):
        b, f = arm(items, sub, a.base, a.key), arm(items, sub, ft, a.key)
        if b is None or f is None:
            report["tasks"][label] = {"error": "missing predictions for one arm"}
            continue
        shared = sorted(set(b) & set(f))
        diffs = np.asarray([f[v] for v in shared]) - np.asarray([b[v] for v in shared])
        try:
            wp = float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
        except ValueError:
            wp = None
        report["tasks"][label] = {
            "n_base_scored": len(b), "n_lora_scored": len(f), "n_paired": len(shared),
            "base_mean_rho": round(float(np.mean([b[v] for v in shared])), 4),
            "lora_mean_rho": round(float(np.mean([f[v] for v in shared])), 4),
            "base_mean_rho_all": round(float(np.mean(list(b.values()))), 4),
            "lora_mean_rho_all": round(float(np.mean(list(f.values()))), 4),
            "base_ci95_boot": boot_mean_ci([b[v] for v in shared]),
            "lora_ci95_boot": boot_mean_ci([f[v] for v in shared]),
            "delta_mean": round(float(np.mean(diffs)), 4),
            "delta_ci95_boot": boot_mean_ci(list(diffs)),
            "wilcoxon_p": (round(wp, 6) if wp is not None else None),
            "n_improved": int((diffs > 0).sum()), "n_worsened": int((diffs < 0).sum()),
            "n_tied": int((diffs == 0).sum()),
        }

    j = report["tasks"].get("T1-JOINT", {})
    report["preregistered_criterion"] = {
        "rule": "T1-JOINT delta mean rho >= 0.10 and Wilcoxon p < 0.05",
        "delta_mean": j.get("delta_mean"), "wilcoxon_p": j.get("wilcoxon_p"),
        "met": bool(j.get("delta_mean") is not None and j["delta_mean"] >= 0.10
                    and j.get("wilcoxon_p") is not None and j["wilcoxon_p"] < 0.05),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
