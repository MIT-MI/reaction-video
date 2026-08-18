"""T2 match: reaction frames + 4 same-video stimulus segments (A-D) -> pick the trigger.

Resumable: predictions append to results/match/<model_slug>.jsonl, items already present are
skipped. Every call is cost-ledgered (see costs.py); the run aborts on the provider cap.

    python -m gold_eval.run_match --model tinker:Qwen/Qwen3.5-9B --limit 1
    python -m gold_eval.run_match --model gemini-3.7-flash
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
LETTERS = "ABCD"

PROMPT = (
    "A viewer is watching a video and reacting to it. First you will see frames (1 per second) "
    "of the VIEWER'S REACTION. Then four candidate STIMULUS segments (A, B, C, D) from the video "
    "they were watching, also as frames. Exactly one segment is what the viewer is reacting to; "
    "the others come from different times in the same video. Which segment triggered this "
    "reaction? Reply with ONLY a single letter (A, B, C, or D) and nothing else."
)


def build_parts(task: dict, r2_base: str, reaction_cap: int, option_cap: int) -> list[dict]:
    parts = [{"type": "text", "text": PROMPT}, {"type": "text", "text": "VIEWER'S REACTION:"}]
    clip = fr.fetch_clip(r2_base, task["reaction_r2_key"])
    for f in fr.extract_frames(clip, max_frames=reaction_cap):
        parts.append({"type": "image_url", "image_url": {"url": fr.data_uri(f)}})
    for i, opt in enumerate(task["options"]):
        parts.append({"type": "text", "text": f"SEGMENT {LETTERS[i]}:"})
        oclip = fr.fetch_clip(r2_base, opt["key"])
        for f in fr.extract_frames(oclip, max_frames=option_cap):
            parts.append({"type": "image_url", "image_url": {"url": fr.data_uri(f)}})
    parts.append({"type": "text", "text": "Answer (single letter A/B/C/D):"})
    return parts


def parse_letter(text: str) -> int | None:
    t = text.strip()
    if t[:1] in LETTERS and (len(t) == 1 or not t[1].isalnum()):
        return LETTERS.index(t[0])
    # case-sensitive so the article "a" never matches; take the LAST standalone letter
    # (models that explain first put the verdict at the end)
    matches = re.findall(r"\b(?:segment\s+)?([ABCD])\b", t)
    return LETTERS.index(matches[-1]) if matches else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", type=Path, default=HERE / "tasks/match_set.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reaction_cap", type=int, default=12, help="max reaction frames")
    p.add_argument("--option_cap", type=int, default=6, help="max frames per option")
    args = p.parse_args()

    payload = json.loads(args.tasks.read_text())
    r2_base, tasks = payload["r2_base"], payload["tasks"]
    out = HERE / "results/match" / f"{slug(args.model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    # resume: an unparsed row (pred_index null) counts as NOT done and is retried;
    # scorers dedupe by id with last-row-wins, so the append-only log stays consistent
    done = {r["candidate_id"] for r in
            (json.loads(l) for l in out.read_text().splitlines() if l.strip())
            if r["pred_index"] is not None} if out.exists() else set()

    backbone = get_backbone(args.model)
    ledger = Ledger(args.model)
    todo = [t for t in tasks if t["candidate_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[match] {args.model}: {len(todo)} to run ({len(done)} done, {len(tasks)} total)")

    for i, task in enumerate(todo):
        cid = task["candidate_id"]
        try:
            parts = call_with_retry(build_parts, task, r2_base, args.reaction_cap, args.option_cap)
            text, tin, tout = call_with_retry(backbone.complete, parts, max_tokens=48)
            usd = ledger.log(cid, tin, tout)
            row = {"candidate_id": cid, "pred_index": parse_letter(text), "raw": text.strip(),
                   "in_tok": tin, "out_tok": tout}
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            mark = "OK " if row["pred_index"] == task["correct_index"] else "err"
            print(f"[{i+1}/{len(todo)}] {cid} pred={row['pred_index']} {mark} "
                  f"(${ledger.provider_spent:.3f} {ledger.provider} total)", flush=True)
        except BudgetExceeded:
            raise
        except Exception:
            print(f"[match] FAIL {cid}\n{traceback.format_exc()}", flush=True)
    print(f"[match] done -> {out}")


if __name__ == "__main__":
    main()
