"""V3 match variant: side-by-side (reaction | candidate) native video, one option per call.

The reaction and each candidate stimulus segment are hstacked into ONE silent video on a
shared timeline, so a video-native model can check temporal alignment directly (do the
viewer's reactions line up with events on the right?). 1-10 score per option, argmax.
Acknowledged caveat: hstacked layouts are out-of-distribution for most models — that OOD
cost vs the alignment-visibility gain is exactly what this arm measures. Gemini only
(native video input).

    python -m gold_eval.run_match_sbs --model gemini-3.7-flash --limit 2
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
    "You will see one video with two panels playing on the same timeline. LEFT: a viewer's "
    "reaction while watching a video. RIGHT: one candidate segment from the video they were "
    "watching. If the right side is the true trigger, the viewer's facial and body reactions "
    "on the left should line up in time with events on the right. Rate from 1 to 10 how "
    "likely it is that the RIGHT side is what the viewer is reacting to. Reply with a single "
    "integer only."
)


def option_parts(task: dict, opt: dict, r2_base: str) -> list[dict]:
    reaction = fr.fetch_clip(r2_base, task["reaction_r2_key"])
    stim = fr.fetch_clip(r2_base, opt["key"])
    sbs = fr.make_sbs(reaction, stim)
    return [{"type": "text", "text": PROMPT}, {"type": "video", "path": str(sbs)},
            {"type": "text", "text": "Score (1-10):"}]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gemini-3.7-flash")
    p.add_argument("--tasks", type=Path, default=HERE / "tasks/match_set.json")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    payload = json.loads(args.tasks.read_text())
    r2_base, tasks = payload["r2_base"], payload["tasks"]
    out = HERE / "results/match_sbs" / f"{slug(args.model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["candidate_id"] for r in
            (json.loads(l) for l in out.read_text().splitlines() if l.strip())
            if r.get("scores") and None not in r["scores"]} if out.exists() else set()

    backbone, ledger = get_backbone(args.model), Ledger(args.model)
    todo = [t for t in tasks if t["candidate_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[m-sbs] {args.model}: {len(todo)} to run ({len(done)} done, {len(tasks)} total)")

    for i, task in enumerate(todo):
        cid = task["candidate_id"]
        try:
            scores = []
            for opt in task["options"]:
                parts = call_with_retry(option_parts, task, opt, r2_base)
                text, tin, tout = call_with_retry(backbone.complete, parts, max_tokens=8)
                ledger.log(f"{cid}:{opt['key']}", tin, tout)
                scores.append(parse_int(text, 1, 10))
            valid = [s for s in scores if s is not None]
            best = max(valid) if valid else None
            tie = best is not None and scores.count(best) > 1
            pred = scores.index(best) if best is not None and not tie else None
            row = {"candidate_id": cid, "scores": scores, "pred_index": pred, "tie": tie}
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            mark = "OK " if pred == task["correct_index"] else ("tie" if tie else "err")
            print(f"[{i+1}/{len(todo)}] {cid} scores={scores} {mark} "
                  f"(${ledger.provider_spent:.3f})", flush=True)
        except BudgetExceeded:
            raise
        except Exception:
            print(f"[m-sbs] FAIL {cid}\n{traceback.format_exc()}", flush=True)
    print(f"[m-sbs] done -> {out}")


if __name__ == "__main__":
    main()
