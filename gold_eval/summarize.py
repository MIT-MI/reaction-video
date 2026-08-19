"""Aggregate every result file into one master summary (results/summary.json + markdown).

Covers: T2 match (single-call MCQ, per-option scored, side-by-side, native video),
T1 ranking (joint full-video, independent per-moment, native video), T3 rationale
(cross-family judge, judge-judge agreement on the overlap slice), human ceilings.

    python -m gold_eval.summarize [--with_ceiling]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, wilcoxon

from .costs import slug  # noqa: F401  (kept for symmetry with other modules)
from .judge_rationale import DIMS, aggregate
from .score_match import score as match_score
from .score_ranking import human_ceiling, per_video_rhos, score_model

RNG_SEED, N_BOOT = 0, 10_000


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson interval for a binomial proportion (G2)."""
    if n == 0:
        return None
    p, zz = k / n, z * z
    denom = 1 + zz / n
    centre = (p + zz / (2 * n)) / denom
    hw = z * math.sqrt(p * (1 - p) / n + zz / (4 * n * n)) / denom
    return [round(centre - hw, 4), round(centre + hw, 4)]


def boot_mean_ci(vals: list[float]) -> list[float] | None:
    """95% percentile bootstrap CI of the mean, seeded (G2)."""
    if len(vals) < 5:
        return None
    rng = np.random.default_rng(RNG_SEED)
    arr = np.asarray(vals)
    means = rng.choice(arr, size=(N_BOOT, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]

HERE = Path(__file__).parent
RES = HERE / "results"


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def dedup(rs: list[dict], key: str) -> dict[str, dict]:
    return {r[key]: r for r in rs}  # last row wins


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--with_ceiling", action="store_true", help="recompute human LOO ceiling (Supabase)")
    args = p.parse_args()

    match_tasks = json.loads((HERE / "tasks/match_set.json").read_text())["tasks"]
    ranking_items = json.loads((HERE / "tasks/ranking_set.json").read_text())
    summary: dict = {"match": {}, "ranking": {}, "rationale": {}}

    for sub in ("match", "match_scored", "match_sbs"):
        d = RES / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.jsonl")):
            preds = dedup(rows(f), "candidate_id")
            s = match_score(match_tasks, preds)
            if any("tie" in r for r in preds.values()):
                ties = sum(1 for r in preds.values() if r.get("tie"))
                s["tie_rate"] = round(ties / len(preds), 4)
            if s["accuracy"] is not None:
                s["acc_ci95_wilson"] = wilson_ci(round(s["accuracy"] * s["n_scored"]), s["n_scored"])
            summary["match"][f"{sub}/{f.stem}"] = s

    # E1 paired test: frames vs native video on the same items (same model) — McNemar exact
    summary["match_paired"] = {}
    from scipy.stats import binomtest
    for base in [k for k in summary["match"] if k.endswith("__video")]:
        frames_key = base[: -len("__video")]
        if frames_key not in summary["match"]:
            continue
        fp = dedup(rows(RES / (frames_key + ".jsonl")), "candidate_id")
        vp = dedup(rows(RES / (base + ".jsonl")), "candidate_id")
        gt = {t["candidate_id"]: t["correct_index"] for t in match_tasks}
        shared = [c for c in gt if c in fp and c in vp]
        b = sum(1 for c in shared if fp[c]["pred_index"] == gt[c] and vp[c]["pred_index"] != gt[c])
        c_ = sum(1 for c in shared if fp[c]["pred_index"] != gt[c] and vp[c]["pred_index"] == gt[c])
        summary["match_paired"][frames_key.split("/", 1)[1]] = {
            "n_paired": len(shared), "frames_only_correct": b, "video_only_correct": c_,
            "mcnemar_p": round(float(binomtest(min(b, c_), b + c_, 0.5).pvalue), 4)
            if b + c_ > 0 else None}

    rho_maps: dict[str, dict[str, float]] = {}
    for sub in ("ranking", "ranking_indep"):
        d = RES / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.jsonl")):
            preds = dedup(rows(f), "video_id")
            s = score_model(ranking_items, preds)
            rho_maps[f"{sub}/{f.stem}"] = per_video_rhos(ranking_items, preds)
            s["mean_spearman_ci95_boot"] = boot_mean_ci(list(rho_maps[f"{sub}/{f.stem}"].values()))
            summary["ranking"][f"{sub}/{f.stem}"] = s
    if args.with_ceiling:
        summary["ranking"]["human_ceiling"] = human_ceiling(ranking_items)

    # G1: paired joint-vs-independent comparison on each model's video intersection
    summary["ranking_paired"] = {}
    for joint_key in [k for k in rho_maps if k.startswith("ranking/") and "__video" not in k]:
        indep_key = joint_key.replace("ranking/", "ranking_indep/")
        if indep_key not in rho_maps:
            continue
        j, ind = rho_maps[joint_key], rho_maps[indep_key]
        shared = sorted(set(j) & set(ind))
        if len(shared) < 5:
            continue
        dj, di = [j[v] for v in shared], [ind[v] for v in shared]
        diffs = np.asarray(di) - np.asarray(dj)
        try:
            wp = float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
        except ValueError:
            wp = None
        summary["ranking_paired"][joint_key.split("/", 1)[1]] = {
            "n_paired": len(shared),
            "joint_mean_rho": round(float(np.mean(dj)), 4),
            "indep_mean_rho": round(float(np.mean(di)), 4),
            "delta_mean": round(float(np.mean(diffs)), 4),
            "delta_ci95_boot": boot_mean_ci(list(diffs)),
            "wilcoxon_p": (round(wp, 6) if wp is not None else None),
        }

    jd = RES / "rationale_judge"
    if jd.is_dir():
        for f in sorted(jd.glob("*__by__*.jsonl")):
            agg = aggregate(f)
            comps = [sum(r["scores"].values()) for r in dedup(rows(f), "candidate_id").values()
                     if r.get("scores")]
            agg["composite_ci95_boot"] = boot_mean_ci(comps)
            summary["rationale"][f.stem] = agg
        # judge-judge agreement on the overlap slice
        gens = sorted({f.stem.split("__by__")[0] for f in jd.glob("*__by__*.jsonl")})
        agree = {}
        for g in gens:
            files = sorted(jd.glob(f"{g}__by__*.jsonl"))
            if len(files) < 2:
                continue
            a, b = (dedup([r for r in rows(f) if r.get("scores")], "candidate_id") for f in files[:2])
            shared = sorted(set(a) & set(b))
            if len(shared) < 5:
                continue
            ca = [sum(a[c]["scores"].values()) for c in shared]
            cb = [sum(b[c]["scores"].values()) for c in shared]
            agree[g] = {"n": len(shared),
                        "pearson": round(float(pearsonr(ca, cb).statistic), 3)
                        if len(set(ca)) > 1 and len(set(cb)) > 1 else None,
                        "mean_abs_diff": round(float(np.mean(np.abs(np.array(ca) - np.array(cb)))), 3)}
        summary["rationale"]["judge_agreement_overlap"] = agree

    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    for section, entries in summary.items():
        print(f"\n## {section}")
        for name, s in entries.items():
            print(f"  {name}: {json.dumps(s)}")
    print(f"\n-> {RES / 'summary.json'}")


if __name__ == "__main__":
    main()
