"""E3 W/O-context ablation for T1: independent per-moment intensity, no full-video context.

Each moment is scored alone from its window frames (same 0-100 behavioral anchors as the
joint prompt in run_ranking.py). Comparing this against the joint full-video run separates
absolute perception from within-video relative calibration — the honest version of the old
paper's Fig-4 context ablation. Output rows share run_ranking's schema so score_ranking.py
works on the directory unchanged.

    python -m gold_eval.run_ranking_independent --model gpt-5 --limit 2
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from . import frames as fr
from .backbones import get_backbone
from .costs import BudgetExceeded, Ledger, slug
from .retry import call_with_retry
from .run_match_scored import parse_int

HERE = Path(__file__).parent

PROMPT = (
    "You will see frames (1 per second) of a viewer's reaction during ONE moment of a video "
    "they are watching. Rate the INTENSITY of the viewer's reaction on a 0-100 scale (0 = "
    "calm baseline watching, 100 = peak reaction such as screaming, jumping, hands on face). "
    "Reply with a single integer only."
)


def moment_parts(item: dict, m: dict, cap: int) -> list[dict]:
    r2_base = item["face_url"][: -len(item["face_r2_key"])]
    clip = fr.fetch_clip(r2_base, item["face_r2_key"])
    parts = [{"type": "text", "text": PROMPT}]
    for f in fr.extract_frames(clip, max_frames=cap, start_s=m["start_s"], end_s=m["end_s"]):
        parts.append({"type": "image_url", "image_url": {"url": fr.data_uri(f)}})
    parts.append({"type": "text", "text": "Score (0-100):"})
    return parts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", type=Path, default=HERE / "tasks/ranking_set.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--window_cap", type=int, default=12)
    args = p.parse_args()

    items = json.loads(args.tasks.read_text())
    out = HERE / "results/ranking_indep" / f"{slug(args.model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["video_id"] for r in
            (json.loads(l) for l in out.read_text().splitlines() if l.strip())
            if r.get("scores")} if out.exists() else set()

    backbone, ledger = get_backbone(args.model), Ledger(args.model)
    todo = [it for it in items if it["video_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[r-indep] {args.model}: {len(todo)} to run ({len(done)} done, {len(items)} total)")

    for i, item in enumerate(todo):
        vid = item["video_id"]
        try:
            scores = {}
            for j, m in enumerate(item["moments"]):
                parts = call_with_retry(moment_parts, item, m, args.window_cap)
                text, tin, tout = call_with_retry(backbone.complete, parts, max_tokens=8)
                ledger.log(f"{vid}:m{j}", tin, tout)
                v = parse_int(text, 0, 100)
                if v is None:
                    raise ValueError(f"unparsable score for m{j}: {text!r}")
                scores[f"m{j}"] = float(v)
            row = {"video_id": vid, "scores": scores}
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{i+1}/{len(todo)}] {vid} {scores} (${ledger.provider_spent:.3f})", flush=True)
        except BudgetExceeded:
            raise
        except Exception:
            print(f"[r-indep] FAIL {vid}\n{traceback.format_exc()}", flush=True)
    print(f"[r-indep] done -> {out}")


if __name__ == "__main__":
    main()
