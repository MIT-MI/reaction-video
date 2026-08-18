"""E2-V1 match variant: per-option scoring (the old paper's Task 1.1 protocol).

One call per option: reaction frames + ONE candidate stimulus segment -> 1-10 likelihood.
Prediction = argmax over the four scores; a non-unique argmax counts as a tie (scored as
incorrect, tie rate reported) — this bridges the old protocol to the new single-call MCQ and
tests whether multi-image interleaving handicaps some models.

    python -m gold_eval.run_match_scored --model tinker:Qwen/Qwen3.6-35B-A3B --limit 2
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

from . import frames as fr
from .backbones import get_backbone
from .costs import BudgetExceeded, Ledger, slug
from .retry import call_with_retry

HERE = Path(__file__).parent

PROMPT = (
    "A viewer is watching a video. You will see frames (1 per second) of the VIEWER'S "
    "REACTION, then frames of ONE candidate segment from the video they were watching. "
    "Rate from 1 to 10 how likely it is that THIS segment is what the viewer is reacting "
    "to at this moment. Reply with a single integer only."
)


def option_parts(task: dict, opt: dict, r2_base: str, reaction_cap: int, option_cap: int) -> list[dict]:
    parts = [{"type": "text", "text": PROMPT}, {"type": "text", "text": "VIEWER'S REACTION:"}]
    clip = fr.fetch_clip(r2_base, task["reaction_r2_key"])
    for f in fr.extract_frames(clip, max_frames=reaction_cap):
        parts.append({"type": "image_url", "image_url": {"url": fr.data_uri(f)}})
    parts.append({"type": "text", "text": "CANDIDATE SEGMENT:"})
    oclip = fr.fetch_clip(r2_base, opt["key"])
    for f in fr.extract_frames(oclip, max_frames=option_cap):
        parts.append({"type": "image_url", "image_url": {"url": fr.data_uri(f)}})
    parts.append({"type": "text", "text": "Score (1-10):"})
    return parts


def parse_int(text: str, lo: int, hi: int) -> int | None:
    m = re.search(r"\b(\d{1,3})\b", text)
    if not m:
        return None
    v = int(m.group(1))
    return v if lo <= v <= hi else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", type=Path, default=HERE / "tasks/match_set.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reaction_cap", type=int, default=12)
    p.add_argument("--option_cap", type=int, default=6)
    args = p.parse_args()

    payload = json.loads(args.tasks.read_text())
    r2_base, tasks = payload["r2_base"], payload["tasks"]
    out = HERE / "results/match_scored" / f"{slug(args.model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["candidate_id"] for r in
            (json.loads(l) for l in out.read_text().splitlines() if l.strip())
            if r.get("scores") and None not in r["scores"]} if out.exists() else set()

    backbone, ledger = get_backbone(args.model), Ledger(args.model)
    todo = [t for t in tasks if t["candidate_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[m-scored] {args.model}: {len(todo)} to run ({len(done)} done, {len(tasks)} total)")

    for i, task in enumerate(todo):
        cid = task["candidate_id"]
        try:
            scores = []
            for opt in task["options"]:
                parts = call_with_retry(option_parts, task, opt, r2_base,
                                        args.reaction_cap, args.option_cap)
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
            print(f"[m-scored] FAIL {cid}\n{traceback.format_exc()}", flush=True)
    print(f"[m-scored] done -> {out}")


if __name__ == "__main__":
    main()
