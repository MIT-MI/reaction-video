"""Build SFT samples for the intensity fine-tune (finetune/PLAN.md §3).

Reads the dataset-repo split (train/val.jsonl), cuts frames from the R2 face videos with the
SAME extractor the eval harness uses, and writes chat-format JSONL with image paths:
  F-INDEP: [window frames] + E3 prompt -> "<int 0-100>"
  F-JOINT: [<=32 full-video frames] + T1 joint prompt -> {"m0": int, ...}
Prompts are imported from the eval runners so train/eval strings cannot drift.

    python -m finetune.build_sft_data --split ../reaction_video_benchmark_phase0_pilot/reports/phase0/finetune --limit 5
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from gold_eval import frames as fr
from gold_eval.run_match import LETTERS, PROMPT as MATCH_PROMPT
from gold_eval.run_ranking import PROMPT as JOINT_PROMPT
from gold_eval.run_ranking_independent import PROMPT as INDEP_PROMPT

HERE = Path(__file__).parent
OUT = HERE / "data"


def to100(x: float) -> int:
    return int(round(x / 3.0 * 100))


def indep_sample(item: dict, m: dict, cap: int, label_key: str) -> dict:
    r2_base = item["face_url"][: -len(item["face_r2_key"])]
    clip = fr.fetch_clip(r2_base, item["face_r2_key"])
    imgs = [str(p) for p in fr.extract_frames(clip, max_frames=cap, start_s=m["start_s"], end_s=m["end_s"])]
    # part order mirrors run_ranking_independent: prompt -> frames -> "Score (0-100):"
    return {"id": f"indep::{m['candidate_id']}", "format": "indep", "images": imgs,
            "prompt_before": INDEP_PROMPT, "prompt": "Score (0-100):",
            "answer": str(to100(m[label_key])), "weight": 1.0 / (1.0 + m["std"])}


def joint_sample(item: dict, cap: int, label_key: str) -> dict:
    r2_base = item["face_url"][: -len(item["face_r2_key"])]
    clip = fr.fetch_clip(r2_base, item["face_r2_key"])
    imgs = [str(p) for p in fr.extract_frames(clip, max_frames=cap)]
    moments_txt = "\n".join(f'  m{i}: seconds {m["start_s"]:.1f} to {m["end_s"]:.1f}'
                            for i, m in enumerate(item["moments"]))
    prompt = JOINT_PROMPT.format(k=len(item["moments"]), moments=moments_txt) + "\nJSON scores:"
    ans = json.dumps({f"m{i}": to100(m[label_key]) for i, m in enumerate(item["moments"])})
    # part order mirrors run_ranking: prompt(+moments) -> frames -> "JSON scores:"
    return {"id": f"joint::{item['video_id']}", "format": "joint", "images": imgs,
            "prompt_before": prompt[: -len("\nJSON scores:")], "prompt": "JSON scores:",
            "answer": ans, "weight": 1.0}


def match_sample(task: dict, r2_base: str, reaction_cap: int = 12, option_cap: int = 6) -> dict:
    """F-MATCH: interleaved parts exactly as run_match.build_parts lays them out; answer = letter.
    Labels are auto temporal alignment (weak supervision, disclosed)."""
    parts = [{"type": "text", "text": MATCH_PROMPT}, {"type": "text", "text": "VIEWER'S REACTION:"}]
    clip = fr.fetch_clip(r2_base, task["reaction_r2_key"])
    parts += [{"type": "image", "path": str(f)} for f in fr.extract_frames(clip, max_frames=reaction_cap)]
    for i, opt in enumerate(task["options"]):
        parts.append({"type": "text", "text": f"SEGMENT {LETTERS[i]}:"})
        oclip = fr.fetch_clip(r2_base, opt["key"])
        parts += [{"type": "image", "path": str(f)} for f in fr.extract_frames(oclip, max_frames=option_cap)]
    parts.append({"type": "text", "text": "Answer (single letter A/B/C/D):"})
    return {"id": f"match::{task['candidate_id']}", "format": "match", "parts": parts,
            "answer": LETTERS[task["correct_index"]], "weight": 1.0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True, help="dir with train.jsonl / val.jsonl")
    p.add_argument("--label", default="debiased", choices=["debiased", "mean"])
    p.add_argument("--indep_cap", type=int, default=12)
    p.add_argument("--joint_cap", type=int, default=32)
    p.add_argument("--joint_oversample", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="videos per split (smoke test)")
    p.add_argument("--match_train", type=Path, default=None,
                   help="match_train.json from build_match_tasks --train_out (adds F-MATCH rows to train; "
                        "every 10th item by hashed id goes to val)")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    match_rows = {"train": [], "val": []}
    if a.match_train:
        import hashlib
        payload = json.loads(a.match_train.read_text())
        for t in payload["tasks"][: a.limit] if a.limit else payload["tasks"]:
            try:
                row = match_sample(t, payload["r2_base"])
            except Exception as e:
                print(f"[sft] FAIL match {t['candidate_id']}: {e}"); continue
            split = "val" if int(hashlib.sha1(t["candidate_id"].encode()).hexdigest(), 16) % 10 == 0 else "train"
            match_rows[split].append(row)
        print(f"[sft] match: train={len(match_rows['train'])} val={len(match_rows['val'])}")
    for name in ("train", "val"):
        items = [json.loads(l) for l in (a.split / f"{name}.jsonl").read_text().splitlines() if l.strip()]
        if a.limit:
            items = items[: a.limit]
        n_i = n_j = n_fail = 0
        with open(OUT / f"{name}.jsonl", "w") as f:
            for it in items:
                try:
                    for m in it["moments"]:
                        f.write(json.dumps(indep_sample(it, m, a.indep_cap, a.label)) + "\n"); n_i += 1
                    if len(it["moments"]) >= 2:
                        s = joint_sample(it, a.joint_cap, a.label)
                        for _ in range(a.joint_oversample if name == "train" else 1):
                            f.write(json.dumps(s) + "\n"); n_j += 1
                except Exception as e:  # one bad video must not kill the build
                    n_fail += 1; print(f"[sft] FAIL {it['video_id']}: {e}")
            for row in match_rows[name]:
                f.write(json.dumps(row) + "\n")
        print(f"[sft] {name}: indep={n_i} joint={n_j} (oversampled) match={len(match_rows[name])} "
              f"failed_videos={n_fail} -> {OUT/name}.jsonl")


if __name__ == "__main__":
    main()
