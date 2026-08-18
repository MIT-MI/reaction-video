"""T3 rationale generation: stimulus window + reaction window -> one-sentence rationale.

Mirrors the human blind-write protocol (gold_stage_c): the model sees the content the viewer
was watching and the viewer's reaction for the same time window, and writes ONE sentence
naming the concrete cue that caused the reaction. The Qwen llm_draft never enters any prompt
(it is itself a prediction under evaluation).

Judging happens separately (judge_rationale.py, cross-family per ADR-0001/D5).

    python -m gold_eval.run_rationale --model gpt-5 --limit 2
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

HERE = Path(__file__).parent

PROMPT = (
    "A viewer is watching a video. You will see two sets of frames sampled at 1 frame per "
    "second over the same time window: first the CONTENT the viewer was watching, then the "
    "VIEWER'S REACTION to it. In ONE sentence, state what specifically in the content caused "
    "this reaction — name the concrete moment, action, event, or line, not a generic "
    "description of the content. Reply with that single sentence only."
)


def build_parts(task: dict, r2_base: str, cap: int) -> list[dict]:
    parts = [{"type": "text", "text": PROMPT}, {"type": "text", "text": "CONTENT (stimulus):"}]
    stim = fr.fetch_clip(r2_base, task["stimuli_r2_key"])
    for f in fr.extract_frames(stim, max_frames=cap, start_s=task["start_s"], end_s=task["end_s"]):
        parts.append({"type": "image_url", "image_url": {"url": fr.data_uri(f)}})
    parts.append({"type": "text", "text": "VIEWER'S REACTION:"})
    face = fr.fetch_clip(r2_base, task["face_r2_key"])
    for f in fr.extract_frames(face, max_frames=cap, start_s=task["start_s"], end_s=task["end_s"]):
        parts.append({"type": "image_url", "image_url": {"url": fr.data_uri(f)}})
    parts.append({"type": "text", "text": "Your one-sentence explanation:"})
    return parts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", type=Path, default=HERE / "tasks/rationale_set.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--window_cap", type=int, default=12, help="max frames per stream")
    args = p.parse_args()

    payload = json.loads(args.tasks.read_text())
    r2_base, tasks = payload["r2_base"], payload["tasks"]
    out = HERE / "results/rationale" / f"{slug(args.model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["candidate_id"] for r in
            (json.loads(l) for l in out.read_text().splitlines() if l.strip())
            if r.get("rationale")} if out.exists() else set()

    backbone = get_backbone(args.model)
    ledger = Ledger(args.model)
    todo = [t for t in tasks if t["candidate_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[rat] {args.model}: {len(todo)} to run ({len(done)} done, {len(tasks)} total)")

    for i, task in enumerate(todo):
        cid = task["candidate_id"]
        try:
            parts = call_with_retry(build_parts, task, r2_base, args.window_cap)
            text, tin, tout = call_with_retry(backbone.complete, parts, max_tokens=80)
            ledger.log(cid, tin, tout)
            row = {"candidate_id": cid, "rationale": " ".join(text.split()),
                   "in_tok": tin, "out_tok": tout}
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{i+1}/{len(todo)}] {cid} '{row['rationale'][:60]}' "
                  f"(${ledger.provider_spent:.3f} {ledger.provider} total)", flush=True)
        except BudgetExceeded:
            raise
        except Exception:
            print(f"[rat] FAIL {cid}\n{traceback.format_exc()}", flush=True)
    print(f"[rat] done -> {out}")


if __name__ == "__main__":
    main()
