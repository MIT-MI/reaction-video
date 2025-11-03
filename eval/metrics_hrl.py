PROMPT_HRL_S = """
Previous Info: {}

Predict the reaction:
"""


PROMPT_HRL_F = """
Previous Info: {}

Predict the reaction
"""




# eval_hrl_min.py
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from tqdm import tqdm
import argparse
import csv
import math
from collections import Counter

# ---------------------- similarity metric (single function) ----------------------
def text_similarity(a: str, b: str) -> float:
    # TODO: replace with proper text similarity metric
    return 0

# ----------------------------- batch inference stub -----------------------------
def inference_fn(prompts: List[str]) -> List[str]:
    # TODO: replace with vLLM:
    # from vllm import LLM, SamplingParams
    # llm = LLM("meta-llama/Llama-3-8B-Instruct")
    # outs = llm.generate(prompts, SamplingParams(max_tokens=256))
    # return [o.outputs[0].text.strip() for o in outs]
    return prompts  # echo for wiring

# ------------------------------- data utilities --------------------------------
def load_video_segments(csv_dir: Path) -> Dict[str, List[Dict[str, any]]]:
    segs = {}
    for f in tqdm(sorted(csv_dir.glob("*.csv")), desc="Loading CSVs"):
        df = pd.read_csv(f)
        for r in df.itertuples(index=False):
            video_id = str(r.video).split(".")[0]
            data_item = {
                "start_time_s": float(r.start_time_s),
                "end_time_s": float(r.end_time_s),
                "description": str(r.description)
            }
            segs.setdefault(video_id, []).append(data_item)
    for vid in segs:
        segs[vid].sort(key=lambda x: x["start_time_s"])
    return segs

# ------------------------------- HRL-F (full) -----------------------------------
def build_hrl_f_items(segments: Dict[str, List[Dict[str, float]]], horizon_sec: float) -> List[Tuple[str, str, str]]:
    items = []
    for vid, segs in segments.items():
        if not segs: continue
        T = segs[-1][1]
        if T <= horizon_sec: continue
        cutoff = T - horizon_sec
        past = " ".join(d for s,e,d in segs if e <= cutoff) or ""
        target = " ".join(d for s,e,d in segs if s >= cutoff) or ""
        if not target: continue
        prompt = (
            f"You are given reactions up to {cutoff:.2f}s for video {vid}. "
            f"Predict reactions for ({cutoff:.2f}s..{T:.2f}s). "
            f"Known: {past}\nPrediction:"
        )
        items.append((vid, prompt, target))
    return items

def evaluate_hrl_f(segments: Dict[str, List[Tuple[float, float, str]]], horizon_sec: float, batch_size: int) -> List[Tuple[str, float]]:
    items = build_hrl_f_items(segments, horizon_sec)
    scores = []
    for i in tqdm(range(0, len(items), batch_size), desc="HRL-F"):
        batch = items[i:i+batch_size]
        preds = inference_fn([p for _, p, _ in batch])
        for (vid, _p, tgt), pred in zip(batch, preds):
            scores.append((vid, text_similarity(pred, tgt)))
    return scores

# ----------------------------- HRL-S (streaming) --------------------------------
def summarize_seg(vid: str, s: float, e: float) -> str:
    return f"[{vid} {s:.2f}-{e:.2f}s]"

def build_hrl_s_items(segments: Dict[str, List[Tuple[float, float, str]]], last_only: bool=True) -> List[Tuple[str, str, str]]:
    items = []
    for vid, segs in segments.items():
        if len(segs) < 2: continue
        idxs = [len(segs)-1] if last_only else list(range(1, len(segs)))
        for n in idxs:
            hist = []
            for i in range(0, n):
                s,e,d = segs[i]
                hist.append(f"Video_{i+1}: {summarize_seg(vid, s, e)}")
                hist.append(f"Reaction_{i+1}: {d}")
            sN,eN,_ = segs[n]
            hist.append(f"Video_{n+1}: {summarize_seg(vid, sN, eN)}")
            prompt = "You will predict the next reaction from streaming context.\n" + "\n".join(hist) + f"\nPredict Reaction_{n+1}:"
            target = segs[n][2]
            items.append((vid, prompt, target))
    return items

def evaluate_hrl_s(segments: Dict[str, List[Tuple[float, float, str]]], batch_size: int, last_only: bool=True) -> List[Tuple[str, float]]:
    items = build_hrl_s_items(segments, last_only)
    scores, acc = [], {}
    for i in tqdm(range(0, len(items), batch_size), desc="HRL-S"):
        batch = items[i:i+batch_size]
        preds = inference_fn([p for _, p, _ in batch])
        for (vid, _p, tgt), pred in zip(batch, preds):
            acc.setdefault(vid, []).append(text_similarity(pred, tgt))
    for vid, vals in acc.items():
        scores.append((vid, sum(vals)/len(vals)))
    return scores

# ------------------------------------ I/O --------------------------------------
def save_scores(path: Path, rows: List[Tuple[str, float]], header=("video_id", "score")):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

# ----------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--out_f_full", default="scores_hrl_f.csv")
    ap.add_argument("--out_f_stream", default="scores_hrl_s.csv")
    ap.add_argument("--horizon_sec", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--stream_last_only", action="store_true")
    args = ap.parse_args()

    segs = load_video_segments(Path(args.csv_dir))
    scores_f = evaluate_hrl_f(segs, args.horizon_sec, args.batch_size)
    save_scores(Path(args.out_f_full), scores_f, header=("video_id", "score_f"))

    scores_s = evaluate_hrl_s(segs, args.batch_size, last_only=args.stream_last_only)
    save_scores(Path(args.out_f_stream), scores_s, header=("video_id", "score_s"))

if __name__ == "__main__":
    main()


