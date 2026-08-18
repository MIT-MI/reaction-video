"""T1 ranking: full face clip at 1 fps + the video's scored moments -> per-moment 0-100 intensity.

The model sees the whole reaction video (frames indexed by second) and the list of moment time
ranges, and returns a JSON object {"m0": score, "m1": score, ...}. Scoring happens later
(score_ranking.py: Spearman vs Stage-A consensus mean, human LOO ceiling alongside).

Resumable; cost-ledgered. prescreen_* fields are never placed in the prompt (ADR-0003).

    python -m gold_eval.run_ranking --model tinker:Qwen/Qwen3.6-35B-A3B --limit 2
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
    "You will see frames of a REACTION VIDEO (a viewer watching content), sampled at 1 frame "
    "per second: frame N corresponds to second N. Below are {k} moments from this video, each a "
    "time range. Rate the INTENSITY of the viewer's reaction in each moment on a 0-100 scale "
    "(0 = calm baseline watching, 100 = peak reaction such as screaming, jumping, hands on "
    "face). Use the full scale to differentiate the moments. Output ONLY a JSON object "
    'mapping moment ids to integer scores, e.g. {{"m0": 40, "m1": 85}}. Do NOT write any '
    "analysis or explanation — the JSON object must be your entire answer.\n\nMoments:\n{moments}"
)


def build_parts(item: dict, frame_cap: int, video: bool = False) -> list[dict]:
    moments_txt = "\n".join(
        f'  m{i}: seconds {m["start_s"]:.1f} to {m["end_s"]:.1f}'
        for i, m in enumerate(item["moments"]))
    prompt = PROMPT.format(k=len(item["moments"]), moments=moments_txt)
    if video:
        prompt = prompt.replace(
            "frames of a REACTION VIDEO (a viewer watching content), sampled at 1 frame per "
            "second: frame N corresponds to second N",
            "a REACTION VIDEO (a viewer watching content)")
    parts = [{"type": "text", "text": prompt}]
    r2_base = item["face_url"][: -len(item["face_r2_key"])]
    clip = fr.fetch_clip(r2_base, item["face_r2_key"])
    if video:
        parts.append({"type": "video", "path": str(clip)})
    else:
        for f in fr.extract_frames(clip, max_frames=frame_cap):
            parts.append({"type": "image_url", "image_url": {"url": fr.data_uri(f)}})
    parts.append({"type": "text", "text": "JSON scores:"})
    return parts


def parse_scores(text: str, k: int) -> dict[str, float] | None:
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for i in range(k):
        v = obj.get(f"m{i}")
        if not isinstance(v, (int, float)):
            return None
        out[f"m{i}"] = float(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", type=Path, default=HERE / "tasks/ranking_set.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--frame_cap", type=int, default=64)
    p.add_argument("--video", action="store_true",
                   help="E1: native mp4 input instead of frames (Gemini only)")
    args = p.parse_args()

    items = json.loads(args.tasks.read_text())
    out = HERE / "results/ranking" / f"{slug(args.model)}{'__video' if args.video else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    # resume: rows whose scores failed to parse count as NOT done and are retried
    done = {r["video_id"] for r in
            (json.loads(l) for l in out.read_text().splitlines() if l.strip())
            if r.get("scores")} if out.exists() else set()

    backbone = get_backbone(args.model)
    ledger = Ledger(args.model)
    todo = [it for it in items if it["video_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[rank] {args.model}: {len(todo)} to run ({len(done)} done, {len(items)} total)")

    for i, item in enumerate(todo):
        vid = item["video_id"]
        try:
            parts = call_with_retry(build_parts, item, args.frame_cap, args.video)
            text, tin, tout = call_with_retry(backbone.complete, parts, max_tokens=320)
            ledger.log(vid, tin, tout)
            scores = parse_scores(text, len(item["moments"]))
            row = {"video_id": vid, "scores": scores, "raw": text.strip()[:400],
                   "in_tok": tin, "out_tok": tout}
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{i+1}/{len(todo)}] {vid} parsed={scores is not None} "
                  f"(${ledger.provider_spent:.3f} {ledger.provider} total)", flush=True)
        except BudgetExceeded:
            raise
        except Exception:
            print(f"[rank] FAIL {vid}\n{traceback.format_exc()}", flush=True)
    print(f"[rank] done -> {out}")


if __name__ == "__main__":
    main()
