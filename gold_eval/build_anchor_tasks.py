"""G3: build the judge-anchor validation set and load it into Supabase.

Samples (generation, item) pairs stratified by the PRIMARY judge's composite score
(low [0,1.5) / mid [1.5,3) / high [3,5]) so the human-judge correlation is estimable across
the whole scale: --per_model per generator x 3 strata (default 4 each -> 12/model -> 84 total
for 7 arms). Deterministic (seeded by ids, no clock/randomness). The annotator UI shows the
rationale + references + clips, never the model name or judge score.

Requires SUPABASE_URL + SUPABASE_SECRET_KEY env (service key; RLS blocks anon inserts on the
tasks table). Deploy schema_anchor.sql in the Supabase SQL editor first.

    python -m gold_eval.build_anchor_tasks --dry_run
    python -m gold_eval.build_anchor_tasks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path

from .costs import slug
from .judge_rationale import judge_for

HERE = Path(__file__).parent
RES = HERE / "results"

GEN_MODELS = ["gpt-5", "gemini-3.7-flash", "tinker:Qwen/Qwen3.5-9B", "tinker:Qwen/Qwen3.6-27B",
              "tinker:Qwen/Qwen3.6-35B-A3B", "tinker:Qwen/Qwen3.5-397B-A17B",
              "tinker:moonshotai/Kimi-K2.6"]
STRATA = [("low", 0.0, 1.5), ("mid", 1.5, 3.0), ("high", 3.0, 5.01)]


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def stable_order(items: list[dict], key: str) -> list[dict]:
    return sorted(items, key=lambda r: hashlib.sha1(r[key].encode()).hexdigest())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per_model", type=int, default=4, help="items per (model, stratum)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    tasks = {t["candidate_id"]: t for t in
             json.loads((HERE / "tasks/rationale_set.json").read_text())["tasks"]}
    refs: dict[str, list[str]] = {}
    for r in json.loads((HERE / "tasks/rationale_refs.json").read_text()):
        refs.setdefault(r["candidate_id"], []).append(r["rationale"])

    out_rows, record = [], Counter()
    for gm in GEN_MODELS:
        gens = {r["candidate_id"]: r for r in rows(RES / "rationale" / f"{slug(gm)}.jsonl")
                if r.get("rationale")}
        judge = judge_for(gm)
        jfile = RES / "rationale_judge" / f"{slug(gm)}__by__{slug(judge)}.jsonl"
        judged = {r["candidate_id"]: sum(r["scores"].values())
                  for r in rows(jfile) if r.get("scores")}
        for name, lo, hi in STRATA:
            pool = [{"candidate_id": c} for c, comp in judged.items()
                    if lo <= comp < hi and c in gens and c in refs and c in tasks]
            take = stable_order(pool, "candidate_id")[: args.per_model]
            record[f"{gm}|{name}"] = len(take)
            for it in take:
                c = it["candidate_id"]
                aid = hashlib.sha1(f"{c}::{gm}".encode()).hexdigest()[:12]
                out_rows.append({
                    "anchor_id": aid, "candidate_id": c, "gen_model": gm,
                    "rationale": gens[c]["rationale"], "refs": refs[c],
                    "face_r2_key": tasks[c]["face_r2_key"],
                    "stimuli_r2_key": tasks[c]["stimuli_r2_key"],
                    "judge_stratum": name,
                })

    print(f"[anchor] sampled {len(out_rows)} pairs; per (model,stratum): {dict(record)}")
    snap = RES / "anchor_set.json"
    snap.write_text(json.dumps(out_rows, indent=1))
    print(f"[anchor] snapshot -> {snap}")
    if args.dry_run:
        return

    url, key = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"]
    req = urllib.request.Request(
        f"{url}/rest/v1/gold_anchor_tasks?on_conflict=anchor_id",
        data=json.dumps(out_rows).encode(),
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates,return=minimal"},
        method="POST")
    urllib.request.urlopen(req)
    print(f"[anchor] upserted {len(out_rows)} gold_anchor_tasks")


if __name__ == "__main__":
    main()
