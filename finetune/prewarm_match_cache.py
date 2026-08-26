"""Pre-fetch and pre-extract the T2 match clips so the eval runs are GPU-bound only.

Uses the same `gold_eval.frames` calls `run_match.build_parts` makes, so it populates exactly
the cache entries the eval will look up. `extract_frames` caches the full 1 fps extraction and
subsamples in Python, so the cached dir does not depend on reaction_cap/option_cap.

    python -m finetune.prewarm_match_cache
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gold_eval import frames as fr

HERE = Path(__file__).parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, default=HERE.parent / "gold_eval/tasks/match_set.json")
    a = p.parse_args()

    payload = json.loads(a.tasks.read_text())
    tasks = payload["tasks"]
    r2_base = payload.get("r2_base") or tasks[0]["reaction_url"][: -len(tasks[0]["reaction_r2_key"])]

    keys: list[str] = []
    for t in tasks:
        keys.append(t["reaction_r2_key"])
        keys += [o["key"] for o in t["options"]]
    keys = list(dict.fromkeys(keys))
    print(f"[prewarm] {len(tasks)} tasks -> {len(keys)} distinct clips")

    ok = failed = 0
    for i, k in enumerate(keys, 1):
        try:
            fr.extract_frames(fr.fetch_clip(r2_base, k))
            ok += 1
        except Exception as e:  # one bad clip must not kill the prewarm
            failed += 1
            print(f"[prewarm] FAIL {k}: {e}")
        if i % 25 == 0 or i == len(keys):
            print(f"[prewarm] {i}/{len(keys)}  ok={ok} failed={failed}", flush=True)
    print(f"[prewarm] done: ok={ok} failed={failed}")


if __name__ == "__main__":
    main()
