"""Fetch/refresh the frozen gold task sets into gold_eval/tasks/ (committed).

- match_set.json: exported read-only from Supabase gold_match_tasks (built by the dataset repo).
- ranking_set.json: copied from the dataset repo's reports/phase0/ranking/ranking_set.json.

    python -m gold_eval.fetch_tasks
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
TASKS = HERE / "tasks"

SUPABASE_URL = "https://tdpzdnnnrfzooeynwkhm.supabase.co"
SUPABASE_ANON = "sb_publishable_mpfqohT44U4sBtZ7vGviNQ_KHUR1mNf"  # read-only under RLS
R2_PUBLIC = "https://pub-b7ceb3feba53440f9924d5ae45fe6f12.r2.dev/"

DEFAULT_RANKING_SRC = (HERE.parents[1]
                       / "reaction_video_benchmark_phase0_pilot/reports/phase0/ranking/ranking_set.json")


def sb_all(path: str) -> list[dict]:
    out, step = [], 1000
    for frm in range(0, 1_000_000, step):
        req = urllib.request.Request(
            f"{SUPABASE_URL}/rest/v1/{path}",
            headers={"apikey": SUPABASE_ANON, "Authorization": f"Bearer {SUPABASE_ANON}",
                     "Range": f"{frm}-{frm + step - 1}"})
        batch = json.load(urllib.request.urlopen(req))
        out += batch
        if len(batch) < step:
            return out
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranking_src", type=Path, default=DEFAULT_RANKING_SRC)
    args = p.parse_args()
    TASKS.mkdir(parents=True, exist_ok=True)

    rows = sb_all("gold_match_tasks?select=candidate_id,video_id,reaction_r2_key,options,"
                  "correct_index,stim_source,consensus_score,cultural_specificity&order=candidate_id")
    (TASKS / "match_set.json").write_text(json.dumps(
        {"r2_base": R2_PUBLIC, "tasks": rows}, indent=1))
    print(f"[fetch] match_set.json: {len(rows)} tasks")

    rt = sb_all("gold_rationale_tasks?select=candidate_id,video_id,start_s,end_s,face_r2_key,"
                "stimuli_r2_key,llm_draft,consensus_score&order=candidate_id")
    (TASKS / "rationale_set.json").write_text(json.dumps({"r2_base": R2_PUBLIC, "tasks": rt}, indent=1))
    refs = [r for r in sb_all("gold_rationales?select=candidate_id,annotator_id,rationale,status"
                              "&order=candidate_id") if r["status"] == "ok"]
    (TASKS / "rationale_refs.json").write_text(json.dumps(refs, indent=1))
    print(f"[fetch] rationale_set.json: {len(rt)} tasks; rationale_refs.json: {len(refs)} human refs")
    # llm_draft (Qwen3.5-9B seed) is a model PREDICTION to evaluate — it must never enter
    # a generation prompt; run_rationale.py only reads keys/times.

    if args.ranking_src.exists():
        shutil.copy(args.ranking_src, TASKS / "ranking_set.json")
        n = len(json.loads((TASKS / "ranking_set.json").read_text()))
        print(f"[fetch] ranking_set.json: {n} videos (from {args.ranking_src})")
    else:
        print(f"[fetch] WARNING ranking source missing: {args.ranking_src}")


if __name__ == "__main__":
    main()
