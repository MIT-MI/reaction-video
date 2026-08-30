"""G3: score the judge-anchor validation study (human raters vs LLM judge).

Reads:
  - results/anchor_set.json                       (84 anchor items, strata, gen models)
  - results/rationale_judge/{gen}__by__{judge}.jsonl  (judge scores per candidate)
  - a JSON dump of the Supabase gold_anchor_scores table (--scores)

Reports:
  - human-judge correlation (Spearman/Pearson, bootstrap CI) on the 0-5 composite,
    vs the shared DeepSeek judge and the cross-family primary judge
  - human-human agreement (pairwise Spearman, Krippendorff's alpha ordinal)
  - per-dimension agreement, per-stratum means, absolute offset

Usage:
  python3 score_anchor.py --scores /path/to/anchor_scores.json [--exclude claude_review_test]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def _rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):  # average ties
        m = a == v
        ranks[m] = ranks[m].mean()
    return ranks


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(_rank(x), _rank(y))

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SHARED_JUDGE = "tinker:deepseek-ai/DeepSeek-V3.1"
SEED = 20260830  # bootstrap seed (deterministic)


def slug(name: str) -> str:
    return re.sub(r"[:/]", "_", name)


def judge_for(gen_model: str) -> str:
    return "gemini-3.7-flash" if not gen_model.startswith(("tinker", "gemini")) else "gpt-5"


def load_judge(gen_model: str, judge: str) -> dict[str, dict]:
    path = RES / "rationale_judge" / f"{slug(gen_model)}__by__{slug(judge)}.jsonl"
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["candidate_id"]] = r["scores"]
    return out


def composite(s: dict) -> float:
    return s["cue"] + s["link"] + s["faithful"]


def spearman_ci(x, y, n_boot=10000):
    rho = spearman(x, y)
    rng = np.random.default_rng(SEED)
    n = len(x)
    boots = []
    x, y = np.asarray(x, float), np.asarray(y, float)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(x[idx])) < 2 or len(set(y[idx])) < 2:
            continue
        boots.append(spearman(x[idx], y[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, lo, hi


def krippendorff_alpha_ordinal(rows: list[list[float]]) -> float:
    """rows: per item, list of ratings (all raters complete here). Ordinal-distance alpha."""
    vals = sorted({v for r in rows for v in r})
    # counts per value overall
    n_ik = defaultdict(lambda: defaultdict(int))
    for i, r in enumerate(rows):
        for v in r:
            n_ik[i][v] += 1
    n_k = defaultdict(int)
    for i in n_ik:
        for v, c in n_ik[i].items():
            n_k[v] += c
    n_total = sum(n_k.values())

    def delta(a, b):  # ordinal metric distance
        if a == b:
            return 0.0
        i1, i2 = sorted((vals.index(a), vals.index(b)))
        s = sum(n_k[vals[g]] for g in range(i1, i2 + 1)) - (n_k[a] + n_k[b]) / 2
        return s**2

    Do = 0.0
    for i, r in enumerate(rows):
        m = len(r)
        if m < 2:
            continue
        pair = sum(delta(a, b) for a, b in combinations(r, 2))
        Do += pair * 2 / (m - 1)
    Do /= n_total
    De = sum(n_k[a] * n_k[b] * delta(a, b) for a, b in combinations(vals, 2)) * 2
    De /= n_total * (n_total - 1)
    return 1 - Do / De if De else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="JSON dump of gold_anchor_scores")
    ap.add_argument("--exclude", nargs="*", default=["claude_review_test"])
    args = ap.parse_args()

    anchors = json.load(open(RES / "anchor_set.json"))
    raw = json.load(open(args.scores))
    raw = [r for r in raw if r["annotator_id"] not in set(args.exclude)]

    by_item = defaultdict(dict)  # anchor_id -> annotator -> scores
    for r in raw:
        by_item[r["anchor_id"]][r["annotator_id"]] = {
            "cue": r["cue"], "link": r["link"], "faithful": r["faithful"]}
    raters = sorted({r["annotator_id"] for r in raw})
    print(f"[anchor] {len(anchors)} items, {len(raters)} raters: {raters}")
    missing = [a["anchor_id"] for a in anchors if len(by_item[a["anchor_id"]]) < len(raters)]
    if missing:
        print(f"[anchor] WARNING incomplete items: {len(missing)} -> {missing}")

    # assemble aligned arrays
    items = []
    for a in anchors:
        gm, cid = a["gen_model"], a["candidate_id"]
        shared = load_judge(gm, SHARED_JUDGE).get(cid)
        primary = load_judge(gm, judge_for(gm)).get(cid)
        hs = [by_item[a["anchor_id"]][r] for r in raters if r in by_item[a["anchor_id"]]]
        items.append({
            "anchor_id": a["anchor_id"], "stratum": a["judge_stratum"], "gen_model": gm,
            "judge_shared": shared, "judge_primary": primary, "humans": hs,
        })
    items = [it for it in items if it["judge_shared"] and it["judge_primary"] and it["humans"]]
    print(f"[anchor] usable items: {len(items)}")

    h_mean = np.array([np.mean([composite(h) for h in it["humans"]]) for it in items])
    j_shared = np.array([composite(it["judge_shared"]) for it in items])
    j_primary = np.array([composite(it["judge_primary"]) for it in items])

    print("\n=== headline: human mean composite (0-5) vs judge composite ===")
    for name, j in [("shared DeepSeek-V3.1", j_shared), ("primary cross-family", j_primary)]:
        rho, lo, hi = spearman_ci(h_mean, j)
        pear = pearson(h_mean, j)
        print(f"{name:24s} Spearman rho={rho:.3f} [95% CI {lo:.3f}, {hi:.3f}]  Pearson r={pear:.3f}")

    print("\n=== human-human reliability (composite) ===")
    per_rater = {r: np.array([composite(by_item[it["anchor_id"]][r]) for it in items]) for r in raters}
    pair_rhos = []
    for a, b in combinations(raters, 2):
        rho = spearman(per_rater[a], per_rater[b])
        pair_rhos.append(rho)
        print(f"  {a[:8]} vs {b[:8]}: rho={rho:.3f}")
    print(f"  mean pairwise human-human rho = {np.mean(pair_rhos):.3f}")
    alpha = krippendorff_alpha_ordinal([[composite(h) for h in it["humans"]] for it in items])
    print(f"  Krippendorff alpha (ordinal, composite) = {alpha:.3f}")

    print("\n=== judge vs single raters (apples-to-apples with human-human) ===")
    jr = [spearman(per_rater[r], j_shared) for r in raters]
    for r, v in zip(raters, jr):
        print(f"  shared judge vs {r[:8]}: rho={v:.3f}")
    print(f"  mean judge-human rho = {np.mean(jr):.3f}  (compare to human-human {np.mean(pair_rhos):.3f})")

    print("\n=== per-dimension (human mean vs shared judge) ===")
    for dim in ("cue", "link", "faithful"):
        hm = np.array([np.mean([h[dim] for h in it["humans"]]) for it in items])
        jd = np.array([it["judge_shared"][dim] for it in items])
        rho = spearman(hm, jd)
        a_dim = krippendorff_alpha_ordinal([[h[dim] for h in it["humans"]] for it in items])
        print(f"  {dim:8s} rho={rho:.3f}  human alpha={a_dim:.3f}  "
              f"human mean={hm.mean():.2f} judge mean={jd.mean():.2f}")

    print("\n=== per-stratum (defined by primary judge at sampling time) ===")
    for st in ("low", "mid", "high"):
        sel = [i for i, it in enumerate(items) if it["stratum"] == st]
        print(f"  {st:4s} n={len(sel):2d}  human mean={h_mean[sel].mean():.2f}  "
              f"shared judge mean={j_shared[sel].mean():.2f}  primary judge mean={j_primary[sel].mean():.2f}")

    print("\n=== absolute offset (composite 0-5) ===")
    print(f"  human mean={h_mean.mean():.3f}  shared judge={j_shared.mean():.3f} "
          f"(judge - human = {j_shared.mean() - h_mean.mean():+.3f})  primary judge={j_primary.mean():.3f}")

    out = {
        "n_items": len(items), "raters": raters,
        "spearman_shared": spearman_ci(h_mean, j_shared),
        "spearman_primary": spearman_ci(h_mean, j_primary),
        "human_human_mean_rho": float(np.mean(pair_rhos)),
        "judge_human_mean_rho": float(np.mean(jr)),
        "alpha_composite": alpha,
    }
    (RES / "anchor_validation.json").write_text(json.dumps(out, indent=2))
    print(f"\n[anchor] wrote {RES / 'anchor_validation.json'}")


if __name__ == "__main__":
    main()
