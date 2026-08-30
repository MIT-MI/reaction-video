"""Reviewer-robustness analyses for the IP (Intensity Profiling / T1 ranking) task.

Addresses two anticipated reviews:
  Q1 — creator concentration: 200/250 videos come from just two channels (four
       creator personas), so video-level CIs may be too tight. We report
       creator-clustered bootstrap CIs (A1) and leave-one-creator-out means (A2),
       at both creator (7 clusters) and channel (5 clusters) granularity.
  Q2 — Spearman comparability: per-video Spearman over mostly 2-3 moments is a
       coarse statistic. We report moment-count-stratified means (A3) and a
       tie-aware pairwise ordering accuracy (A4, chance = 0.5) that is directly
       comparable between models and the human leave-one-out reference.
Plus human-reference re-weighting (A5) and a selection-dependency check (A6).

Scoring conventions. Two variants of score_ranking.per_video_rhos exist:
  strict (tied_as_zero=False): all-tied predictions dropped (rho undefined);
  tie0   (tied_as_zero=True):  all-tied parsed predictions credited rho=0.
Unparsed/missing predictions are dropped under BOTH (never imputed: indep
397B has 24 unparsed videos and the paper's .267 = mean over the 226 parsed,
not a zero-imputed mean over 250). The paper's Table-2 rhos — joint .176/.026/
.029/.163/.043/.065/.086 and indep .316/.177/.255/.329/.210/.267/.233 — are ALL
reproduced to 3 decimals by the tie0 convention (verified; joint gemini raw
0.17552 -> .176), which is therefore marked PRIMARY ("paper_convention") below;
the strict variant is kept as "strict_convention". The two differ only where
all-tied predictions occur (joint 35B; every indep arm).
Spearman is implemented as Pearson on average ranks (scipy-free).

Inputs (all local, no network):
  tasks/ranking_set.json            250 videos with per-moment consensus
  results/ranking/*.jsonl           joint (full-video) predictions
  results/ranking_indep/*.jsonl     independent per-moment predictions
  data/gold_scores_dump.json        per-rater Stage-A scores dump
                                    (candidate_id, annotator_id, score) —
                                    fetched read-only from Supabase gold_scores,
                                    2026-08-30; 14894 rows.

Output: results/ip_robustness.json

Deterministic: seeded numpy Generators only, no wall-clock anywhere.

    python3 score_ip_robustness.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SEED = 0
N_BOOT = 10_000

# The 7 zero-shot arms (finetune checkpoints and __video variants excluded).
MODELS = [
    "gemini-3.7-flash",
    "gpt-5",
    "tinker_Qwen_Qwen3.5-397B-A17B",
    "tinker_Qwen_Qwen3.5-9B",
    "tinker_Qwen_Qwen3.6-27B",
    "tinker_Qwen_Qwen3.6-35B-A3B",
    "tinker_moonshotai_Kimi-K2.6",
]
MODES = {"joint": "results/ranking", "indep": "results/ranking_indep"}

# Paper Table-2 per-video mean Spearman (3 dp) — used only to verify that the
# tie0 convention reproduces the published numbers exactly.
PAPER_TABLE2 = {
    "joint": {"gemini-3.7-flash": 0.176, "gpt-5": 0.026,
              "tinker_Qwen_Qwen3.5-397B-A17B": 0.065, "tinker_Qwen_Qwen3.5-9B": 0.029,
              "tinker_Qwen_Qwen3.6-27B": 0.163, "tinker_Qwen_Qwen3.6-35B-A3B": 0.043,
              "tinker_moonshotai_Kimi-K2.6": 0.086},
    "indep": {"gemini-3.7-flash": 0.316, "gpt-5": 0.177,
              "tinker_Qwen_Qwen3.5-397B-A17B": 0.267, "tinker_Qwen_Qwen3.5-9B": 0.255,
              "tinker_Qwen_Qwen3.6-27B": 0.329, "tinker_Qwen_Qwen3.6-35B-A3B": 0.210,
              "tinker_moonshotai_Kimi-K2.6": 0.233},
}

# Channel-level merge: a reviewer may insist the two personas per channel share
# audience/style, so cluster at channel granularity too (~5 clusters).
CHANNEL_OF = {
    "Magic Magy_person1": "Magic Magy",
    "Magic Magy_person2": "Magic Magy",
    "CineBinge_person1": "CineBinge",
    "CineBinge_person2": "CineBinge",
}

MIN_RATERS = 3      # selection criteria from reports/phase0/ranking/selection_record.json
MIN_SPREAD = 0.25   # (pilot repo build_ranking_tasks.py)


# ---------------------------------------------------------------- primitives

def rank_avg(v: list[float]) -> np.ndarray:
    """Average ranks (1-based), ties get the mean rank — matches scipy rankdata."""
    a = np.asarray(v, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sv = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rho = Pearson on average ranks; nan if either side constant."""
    rx, ry = rank_avg(x), rank_avg(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def rnd(x, k: int = 4):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [rnd(v, k) for v in x]
    return round(float(x), k)


# ---------------------------------------------------------------- data loading

def load_items() -> list[dict]:
    return json.loads((HERE / "tasks/ranking_set.json").read_text())


def load_preds(path: Path) -> dict[str, dict]:
    preds = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            preds[row["video_id"]] = row  # last row wins (summarize.py dedup)
    return preds


def per_video_rhos(items: list[dict], preds: dict[str, dict],
                   tied_as_zero: bool = False) -> dict[str, float]:
    """Exact reproduction of score_ranking.per_video_rhos.

    tied_as_zero=False: strict / pre-registered (all-tied predictions dropped).
    tied_as_zero=True:  paper Table-2 convention (all-tied parsed predictions
    credited rho=0). Unparsed/missing predictions dropped under both."""
    out = {}
    for it in items:
        p = preds.get(it["video_id"])
        if not p or not p.get("scores"):
            continue
        gold = [m["mean"] for m in it["moments"]]
        model = [p["scores"][f"m{i}"] for i in range(len(gold))]
        if len(set(gold)) > 1:
            rho = spearman(model, gold)
            if not math.isnan(rho):
                out[it["video_id"]] = rho
            elif tied_as_zero:
                out[it["video_id"]] = 0.0
    return out


def load_rater_scores(items: list[dict]) -> dict[tuple[str, str], dict[str, int]]:
    """(video_id, annotator_id) -> {candidate_id: score}, restricted to set moments."""
    dump = json.loads((HERE / "data/gold_scores_dump.json").read_text())
    wanted = {m["candidate_id"]: it["video_id"] for it in items for m in it["moments"]}
    by_va: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for row in dump:
        vid = wanted.get(row["candidate_id"])
        if vid:
            by_va[(vid, row["annotator_id"])][row["candidate_id"]] = int(row["score"])
    return by_va


def human_loo(items: list[dict], by_va: dict) -> list[dict]:
    """Leave-one-rater-out records, one per complete (video, rater) pair.

    Exact reproduction of score_ranking.human_ceiling's inclusion rule: the rater
    must have rated every candidate of the video; rho(rater, mean of others);
    pairs with constant rater or constant others-mean (or nan rho) are dropped.
    Each record keeps mine/others_mean so A4 pairwise accuracy reuses them.
    """
    per_video_cands: dict[str, set] = defaultdict(set)
    for (vid, _), scores in by_va.items():
        per_video_cands[vid].update(scores)
    recs = []
    for (vid, ann), scores in sorted(by_va.items()):
        cands = sorted(per_video_cands[vid])
        if len(cands) < 2 or len(scores) < len(cands):
            continue
        others_mean, mine = [], []
        for c in cands:
            others = [s[c] for (v2, a2), s in by_va.items()
                      if v2 == vid and a2 != ann and c in s]
            if not others:
                break
            others_mean.append(statistics.mean(others))
            mine.append(scores[c])
        else:
            if len(set(mine)) > 1 and len(set(others_mean)) > 1:
                rho = spearman(mine, others_mean)
                if not math.isnan(rho):
                    recs.append({"video_id": vid, "annotator_id": ann, "rho": rho,
                                 "mine": mine, "others_mean": others_mean})
    return recs


# ---------------------------------------------------------------- A1 bootstrap

def boot_naive(vals: np.ndarray, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(N_BOOT, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": rnd(vals.mean()), "ci95": rnd([lo, hi]), "ci_width": rnd(hi - lo)}


def boot_clustered(vals_by_cluster: dict[str, list[float]], seed: int = SEED) -> dict:
    """Cluster bootstrap: resample clusters with replacement, mean of pooled units."""
    keys = sorted(vals_by_cluster)
    sums = np.array([sum(vals_by_cluster[k]) for k in keys])
    cnts = np.array([len(vals_by_cluster[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n_clusters": len(keys), "ci95": rnd([lo, hi]), "ci_width": rnd(hi - lo)}


def a1_bootstrap(rho_by_vid: dict[str, float], creator_of: dict[str, str]) -> dict:
    vals = np.array([rho_by_vid[v] for v in sorted(rho_by_vid)])
    naive = boot_naive(vals)
    out = {"n_units": len(vals), "naive": naive}
    for level in ("creator", "channel"):
        by_cl: dict[str, list[float]] = defaultdict(list)
        for v, r in rho_by_vid.items():
            c = creator_of[v]
            by_cl[CHANNEL_OF.get(c, c) if level == "channel" else c].append(r)
        cl = boot_clustered(by_cl)
        cl["width_ratio_vs_naive"] = rnd(cl["ci_width"] / naive["ci_width"])
        out[level] = cl
    return out


# ---------------------------------------------------------------- A2 leave-one-creator-out

def a2_loco(units: list[tuple[str, float]], creator_of: dict[str, str]) -> dict:
    """units = [(video_id, value)]; mean of remaining units when each cluster is held out."""
    out = {}
    for level in ("creator", "channel"):
        cl_of = {v: (CHANNEL_OF.get(creator_of[v], creator_of[v])
                     if level == "channel" else creator_of[v]) for v, _ in units}
        clusters = sorted({cl_of[v] for v, _ in units})
        full = float(np.mean([x for _, x in units]))
        holdout = {}
        for c in clusters:
            rest = [x for v, x in units if cl_of[v] != c]
            holdout[c] = rnd(np.mean(rest)) if rest else None
        vals = [v for v in holdout.values() if v is not None]
        out[level] = {"full_mean": rnd(full), "holdout_means": holdout,
                      "min": rnd(min(vals)), "max": rnd(max(vals)),
                      "range": rnd(max(vals) - min(vals))}
    return out


# ---------------------------------------------------------------- A3 strata

def stratum(n_moments: int) -> str:
    return {2: "2", 3: "3"}.get(n_moments, "4+")


def a3_strata(units: list[tuple[str, float]], nm_of: dict[str, int]) -> dict:
    by_s: dict[str, list[float]] = defaultdict(list)
    for v, x in units:
        by_s[stratum(nm_of[v])].append(x)
    return {s: {"n": len(by_s[s]), "mean": rnd(np.mean(by_s[s]))}
            for s in ("2", "3", "4+") if by_s.get(s)}


# ---------------------------------------------------------------- A4 pairwise accuracy

def pair_acc(pred: list[float], gold: list[float]) -> tuple[float, int]:
    """(sum of pair credits, n non-tied-gold pairs); pred ties earn 0.5."""
    credit, n = 0.0, 0
    for i, j in combinations(range(len(gold)), 2):
        if gold[i] == gold[j]:
            continue
        n += 1
        if pred[i] == pred[j]:
            credit += 0.5
        elif (pred[i] > pred[j]) == (gold[i] > gold[j]):
            credit += 1.0
    return credit, n


def a4_model(items: list[dict], preds: dict[str, dict]) -> dict:
    tot_c = tot_n = 0
    per_video = []
    for it in items:
        p = preds.get(it["video_id"])
        if not p or not p.get("scores"):
            continue
        gold = [m["mean"] for m in it["moments"]]
        model = [p["scores"][f"m{i}"] for i in range(len(gold))]
        c, n = pair_acc(model, gold)
        if n:
            tot_c += c
            tot_n += n
            per_video.append(c / n)
    return {"n_pairs": tot_n, "pooled_acc": rnd(tot_c / tot_n) if tot_n else None,
            "n_videos": len(per_video),
            "video_equal_weight_acc": rnd(np.mean(per_video)) if per_video else None,
            "note": "convention-independent: covers every parsed video with >=1 "
                    "non-tied gold pair (all-tied predictions included, each of "
                    "their pairs credited 0.5); unparsed videos excluded"}


def a4_human(loo_recs: list[dict]) -> dict:
    tot_c = tot_n = 0
    by_video: dict[str, list[float]] = defaultdict(list)
    for r in loo_recs:
        c, n = pair_acc(r["mine"], r["others_mean"])
        if n:
            tot_c += c
            tot_n += n
            by_video[r["video_id"]].append(c / n)
    vid_means = [float(np.mean(v)) for v in by_video.values()]
    return {"n_pairs": tot_n, "pooled_acc": rnd(tot_c / tot_n) if tot_n else None,
            "n_videos": len(vid_means),
            "video_equal_weight_acc": rnd(np.mean(vid_means)) if vid_means else None,
            "note": "LOO: each rater's scores vs mean of other raters; gold ties dropped, "
                    "rater ties credited 0.5; chance = 0.5"}


# ---------------------------------------------------------------- A6 selection dependency

def a6_selection(items: list[dict], by_va: dict) -> dict:
    """For each rater, drop their ratings and re-check each selected video's
    eligibility under the original criteria (>=2 moments with >=3 raters,
    consensus-mean spread >= 0.25). Criteria reconstructed from
    reports/phase0/ranking/selection_record.json (pilot repo)."""
    # candidate -> {annotator: score}
    by_cand: dict[str, dict[str, int]] = defaultdict(dict)
    for (vid, ann), scores in by_va.items():
        for c, s in scores.items():
            by_cand[c][ann] = s
    raters = sorted({ann for (_, ann) in by_va})
    out = {}
    for r in raters:
        keep = 0
        touched = 0
        for it in items:
            means = []
            rated_by_r = False
            for m in it["moments"]:
                sc = {a: s for a, s in by_cand[m["candidate_id"]].items() if a != r}
                if r in by_cand[m["candidate_id"]]:
                    rated_by_r = True
                if len(sc) >= MIN_RATERS:
                    means.append(statistics.mean(sc.values()))
            if rated_by_r:
                touched += 1
            if len(means) >= 2 and (max(means) - min(means)) >= MIN_SPREAD:
                keep += 1
        out[r] = {"videos_rated_in_set": touched,
                  "still_eligible": keep,
                  "frac_still_eligible": rnd(keep / len(items))}
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    items = load_items()
    creator_of = {it["video_id"]: it["creator"] for it in items}
    nm_of = {it["video_id"]: it["n_moments"] for it in items}

    report: dict = {
        "config": {
            "seed": SEED, "n_boot": N_BOOT, "models": MODELS,
            "clusters_creator": sorted({it["creator"] for it in items}),
            "clusters_channel": sorted({CHANNEL_OF.get(it["creator"], it["creator"])
                                        for it in items}),
            "selection_criteria": {"min_raters": MIN_RATERS, "min_spread": MIN_SPREAD},
            "conventions": {
                "paper_convention": "PRIMARY. tied_as_zero: per-video Spearman vs "
                    "consensus mean; all-tied parsed predictions credited rho=0; "
                    "unparsed/missing dropped (NOT zero-imputed). Reproduces every "
                    "paper Table-2 rho (joint AND indep) to 3 decimals — see "
                    "paper_table2_verification.",
                "strict_convention": "SECONDARY. score_ranking pre-registered "
                    "default: all-tied predictions dropped (rho undefined).",
            },
        },
        "paper_table2_verification": {},
        "models": {},
    }

    # ---- per-model analyses (both conventions; paper = primary)
    for mode, sub in MODES.items():
        for m in MODELS:
            f = HERE / sub / f"{m}.jsonl"
            if not f.exists():
                continue
            preds = load_preds(f)
            entry: dict = {}
            raw_means = {}
            for conv, tie0 in (("paper_convention", True), ("strict_convention", False)):
                rhos = per_video_rhos(items, preds, tied_as_zero=tie0)
                units = sorted(rhos.items())
                raw_means[conv] = float(np.mean([x for _, x in units]))
                entry[conv] = {
                    "mean_spearman": rnd(raw_means[conv]),
                    "n_videos_scored": len(units),
                    "A1_bootstrap": a1_bootstrap(rhos, creator_of),
                    "A2_leave_one_creator_out": a2_loco(units, creator_of),
                    "A3_by_n_moments": a3_strata(units, nm_of),
                }
            # n_all_tied_as_zero: count videos dropped by strict but kept by paper
            entry["paper_convention"]["n_all_tied_as_zero"] = (
                entry["paper_convention"]["n_videos_scored"]
                - entry["strict_convention"]["n_videos_scored"])
            entry["A4_pairwise"] = a4_model(items, preds)
            report["models"].setdefault(m, {})[mode] = entry
            expected = PAPER_TABLE2[mode].get(m)
            report["paper_table2_verification"][f"{mode}/{m}"] = {
                "paper": expected,
                "recomputed": rnd(raw_means["paper_convention"], 5),
                "match_3dp": expected is not None
                             and round(raw_means["paper_convention"], 3) == expected}

    # ---- human reference
    dump_path = HERE / "data/gold_scores_dump.json"
    if dump_path.exists():
        by_va = load_rater_scores(items)
        loo = human_loo(items, by_va)
        pair_units = [(r["video_id"], r["rho"]) for r in loo]
        rho_by_pair = {f'{r["video_id"]}::{r["annotator_id"]}': r["rho"] for r in loo}
        creator_of_pair = {f'{r["video_id"]}::{r["annotator_id"]}':
                           creator_of[r["video_id"]] for r in loo}
        vals = np.array([rho_by_pair[k] for k in sorted(rho_by_pair)])

        # A1 (units = rater-video pairs, clustered by the video's creator/channel)
        naive = boot_naive(vals)
        a1 = {"n_units": len(vals), "naive": naive}
        for level in ("creator", "channel"):
            by_cl: dict[str, list[float]] = defaultdict(list)
            for k, r in rho_by_pair.items():
                c = creator_of_pair[k]
                by_cl[CHANNEL_OF.get(c, c) if level == "channel" else c].append(r)
            cl = boot_clustered(by_cl)
            cl["width_ratio_vs_naive"] = rnd(cl["ci_width"] / naive["ci_width"])
            a1[level] = cl

        # A5 re-weighting
        by_video: dict[str, list[float]] = defaultdict(list)
        for r in loo:
            by_video[r["video_id"]].append(r["rho"])
        vid_means = [float(np.mean(v)) for v in by_video.values()]

        report["human_reference"] = {
            "A5_weighting": {
                "paper_convention_mean_over_rater_video_pairs": rnd(vals.mean()),
                "n_rater_video_pairs": len(vals),
                "per_video_first_mean": rnd(np.mean(vid_means)),
                "n_videos_with_loo": len(vid_means),
                "note": "paper convention reproduces the reported 0.649 ceiling",
            },
            "A1_bootstrap": a1,
            "A2_leave_one_creator_out": a2_loco(pair_units, creator_of),
            "A3_by_n_moments": a3_strata(pair_units, nm_of),
            "A4_pairwise": a4_human(loo),
            "A6_selection_dependency": a6_selection(items, by_va),
        }
    else:
        report["human_reference"] = {
            "skipped": "data/gold_scores_dump.json not found; per-rater ratings "
                       "unavailable locally, human-side analyses skipped"}

    out = HERE / "results/ip_robustness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.rename(out)
    print(f"-> {out}")

    # compact console summary (paper convention)
    n_match = sum(1 for v in report["paper_table2_verification"].values() if v["match_3dp"])
    print(f"paper Table-2 verification: {n_match}/{len(report['paper_table2_verification'])} "
          f"match to 3 decimals")
    for m, modes in report["models"].items():
        for mode, e in modes.items():
            pc = e["paper_convention"]
            a1 = pc["A1_bootstrap"]
            print(f"{m} [{mode}] rho={pc['mean_spearman']} (n={pc['n_videos_scored']}, "
                  f"strict={e['strict_convention']['mean_spearman']}) "
                  f"naiveCI={a1['naive']['ci95']} creatorCI={a1['creator']['ci95']} "
                  f"(x{a1['creator']['width_ratio_vs_naive']}) "
                  f"pairacc={e['A4_pairwise']['pooled_acc']}")
    hr = report.get("human_reference", {})
    if "A5_weighting" in hr:
        w = hr["A5_weighting"]
        print(f"human LOO: pair-weighted={w['paper_convention_mean_over_rater_video_pairs']} "
              f"video-weighted={w['per_video_first_mean']} "
              f"pairacc={hr['A4_pairwise']['pooled_acc']}")


if __name__ == "__main__":
    main()
