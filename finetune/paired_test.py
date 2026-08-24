"""Paired base-vs-LoRA test on T1-JOINT and T1-INDEP (PLAN §5 success criterion).

Same statistics as `gold_eval.summarize.ranking_paired`, but pairing the two ARMS of the
fine-tune (local base vs LoRA) on one task instead of the two tasks of one model: per-video
Spearman from `score_ranking.per_video_rhos`, Wilcoxon signed-rank on the differences and a
seeded percentile bootstrap CI of the mean delta (RNG_SEED / N_BOOT imported from summarize so
the numbers match the rest of the report).

Two conventions are reported for every task:

  * **`primary_tie_aware`** (PRIMARY as of run 2) — `per_video_rhos(tied_as_zero=True)` over
    ALL videos with non-tied gold (250). A prediction that assigns the same score to every
    moment of a video is unrankable, so Spearman is undefined; here it is credited rho = 0,
    i.e. chance. Run 1 showed why this has to be primary: the fine-tuned arm emitted all-tied
    scores on 84/250 joint videos against 0 for the base, and the drop-ties convention deletes
    exactly those videos from the FT arm's own average — collapse flattered the fine-tune by
    shrinking its denominator instead of penalising it.
  * **`preregistered_drop_ties`** — the run-1 pre-registered statistic, kept alongside and
    unchanged: drop undefined-Spearman videos, pair over the intersection of what both arms
    scored.

Pre-registered criterion: delta rho on T1-JOINT >= 0.10 with Wilcoxon p < .05. It is evaluated
under both conventions; `preregistered_criterion.met` follows the PRIMARY (tie-aware) one, with
the drop-ties verdict reported next to it.

    python -m finetune.paired_test --lora finetune/ckpt/r2-order-5ep/best \
        --out gold_eval/results/finetune_paired_r2.json
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
TASKS = (("T1-JOINT", "ranking"), ("T1-INDEP", "ranking_indep"))


def preds_for(sub: str, model: str) -> dict[str, dict] | None:
    f = RES / sub / f"{slug(model)}.jsonl"
    if not f.exists():
        return None
    return {r["video_id"]: r for r in
            (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def stats(b_rho: dict[str, float], f_rho: dict[str, float], vids: list[str],
          extra: dict | None = None) -> dict:
    """Wilcoxon + bootstrap CI of the mean delta over `vids` (missing -> 0.0)."""
    B = np.array([b_rho.get(v, 0.0) for v in vids])
    L = np.array([f_rho.get(v, 0.0) for v in vids])
    d = L - B
    try:
        wp = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        wp = None
    return {"n": len(vids),
            "base_mean_rho": round(float(B.mean()), 4) if len(vids) else None,
            "lora_mean_rho": round(float(L.mean()), 4) if len(vids) else None,
            "base_ci95_boot": boot_mean_ci(list(B)),
            "lora_ci95_boot": boot_mean_ci(list(L)),
            "delta_mean": round(float(d.mean()), 4) if len(vids) else None,
            "delta_ci95_boot": boot_mean_ci(list(d)),
            "wilcoxon_p": (round(wp, 6) if wp is not None else None),
            "n_improved": int((d > 0).sum()), "n_worsened": int((d < 0).sum()),
            "n_tied": int((d == 0).sum()), **(extra or {})}


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
    # videos the ranking metric is defined on at all: gold must not be all-tied
    rankable = [it["video_id"] for it in items if len({m[a.key] for m in it["moments"]}) > 1]
    report: dict = {"base": a.base, "lora": ft, "consensus_key": a.key,
                    "primary_convention": "tie_aware (rho=0 credited to all-tied predictions)",
                    "n_videos_total": len(items), "n_videos_nontied_gold": len(rankable),
                    "primary_tie_aware": {}, "preregistered_drop_ties": {}, "coverage": {}}

    for label, sub in TASKS:
        b, f = preds_for(sub, a.base), preds_for(sub, ft)
        if b is None or f is None:
            miss = [nm for nm, p_ in (("base", b), ("lora", f)) if p_ is None]
            err = {"error": f"missing predictions for: {', '.join(miss)}"}
            report["primary_tie_aware"][label] = err
            report["preregistered_drop_ties"][label] = err
            continue

        # ---- PRIMARY: tie-aware over all non-tied-gold videos -------------------------
        bz = per_video_rhos(items, b, key=a.key, tied_as_zero=True)
        fz = per_video_rhos(items, f, key=a.key, tied_as_zero=True)
        # tied_as_zero credits 0 to an all-tied prediction, but a video with NO prediction row
        # (or an unparsed one) is absent entirely; `stats` fills 0.0 for it. Count those so the
        # imputation is auditable rather than silent — expected 0 for a complete eval run.
        report["primary_tie_aware"][label] = stats(
            bz, fz, rankable,
            {"n_absent_filled_zero_base": sum(1 for v in rankable if v not in bz),
             "n_absent_filled_zero_lora": sum(1 for v in rankable if v not in fz)})

        # ---- SECONDARY: the run-1 pre-registered statistic, unchanged ------------------
        bd, fd = per_video_rhos(items, b, key=a.key), per_video_rhos(items, f, key=a.key)
        shared = sorted(set(bd) & set(fd))
        report["preregistered_drop_ties"][label] = stats(
            bd, fd, shared,
            {"n_base_scored": len(bd), "n_lora_scored": len(fd), "n_paired": len(shared)})

        # ---- collapse diagnostics (run 1's "Output-range collapse" table) --------------
        report["coverage"][label] = {
            nm: {"rows": len(p_),
                 "all_tied_predictions": sum(1 for r in p_.values()
                                             if r.get("scores") and len(set(r["scores"].values())) == 1),
                 "unparsed": sum(1 for r in p_.values() if not r.get("scores")),
                 "pred_sd": round(float(np.std([v for r in p_.values() if r.get("scores")
                                                for v in r["scores"].values()])), 2),
                 "pred_mean": round(float(np.mean([v for r in p_.values() if r.get("scores")
                                                   for v in r["scores"].values()])), 2),
                 "n_distinct_values": len({v for r in p_.values() if r.get("scores")
                                           for v in r["scores"].values()}),
                 "n_scored_drop_ties": len(per_video_rhos(items, p_, key=a.key)),
                 "n_scored_tie_aware": len(per_video_rhos(items, p_, key=a.key, tied_as_zero=True))}
            for nm, p_ in (("base", b), ("lora", f))}

    def verdict(block: dict) -> dict:
        j = block.get("T1-JOINT", {})
        return {"delta_mean": j.get("delta_mean"), "wilcoxon_p": j.get("wilcoxon_p"),
                "n": j.get("n"),
                "met": bool(j.get("delta_mean") is not None and j["delta_mean"] >= 0.10
                            and j.get("wilcoxon_p") is not None and j["wilcoxon_p"] < 0.05)}

    prim = verdict(report["primary_tie_aware"])
    report["preregistered_criterion"] = {
        "rule": "T1-JOINT delta mean rho >= 0.10 and Wilcoxon p < 0.05",
        "judged_under": "primary_tie_aware",
        **prim,
        "drop_ties_verdict": verdict(report["preregistered_drop_ties"]),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
