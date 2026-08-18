"""Score T2 match predictions: accuracy, temporal distance of errors, cultural strata.

    python -m gold_eval.score_match                # all models found in results/match/
    python -m gold_eval.score_match --model gemini-3.7-flash
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .costs import slug

HERE = Path(__file__).parent


def score(tasks: list[dict], preds: dict[str, dict]) -> dict:
    n, correct, unparsed, tdists = 0, 0, 0, []
    strata: dict[str, list[int]] = {}
    for t in tasks:
        p = preds.get(t["candidate_id"])
        if p is None:
            continue
        n += 1
        pi = p["pred_index"]
        hit = int(pi == t["correct_index"])
        correct += hit
        cs = str(t.get("cultural_specificity"))
        strata.setdefault(cs, []).append(hit)
        if pi is None:
            unparsed += 1
        elif not hit:
            tdists.append(abs(t["options"][pi]["start_s"] - t["options"][t["correct_index"]]["start_s"]))
    out = {"n_scored": n, "accuracy": round(correct / n, 4) if n else None,
           "chance": 0.25, "unparsed": unparsed,
           "err_temporal_dist_s": {
               "mean": round(statistics.mean(tdists), 2) if tdists else None,
               "median": round(statistics.median(tdists), 2) if tdists else None},
           "acc_by_cultural_specificity": {
               k: round(sum(v) / len(v), 4) for k, v in sorted(strata.items())}}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument("--tasks", type=Path, default=HERE / "tasks/match_set.json")
    args = p.parse_args()

    tasks = json.loads(args.tasks.read_text())["tasks"]
    res_dir = HERE / "results/match"
    files = [res_dir / f"{slug(args.model)}.jsonl"] if args.model else sorted(res_dir.glob("*.jsonl"))
    report = {}
    for f in files:
        if not f.exists():
            print(f"missing: {f}")
            continue
        preds = {json.loads(l)["candidate_id"]: json.loads(l) for l in f.read_text().splitlines() if l.strip()}
        report[f.stem] = score(tasks, preds)
        print(f"{f.stem}: {json.dumps(report[f.stem])}")
    out = res_dir / "scores.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
