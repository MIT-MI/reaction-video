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
import subprocess
import tempfile
from collections import Counter
from models import generate_response

# ---------------------- similarity metric (single function) ----------------------
def reaction_similarity(a: str, b: str) -> float:
    tokens_a = [token for token in a.lower().split() if token]
    tokens_b = [token for token in b.lower().split() if token]
    if not tokens_a or not tokens_b:
        return 0.0

    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)
    dot = sum(counter_a[t] * counter_b[t] for t in counter_a.keys() & counter_b.keys())
    if dot == 0:
        return 0.0

    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

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
        last_end = None
        tmp_vid_segs = []
        should_save = True
        for r in df.itertuples(index=False):
            video_id = str(r.video).split(".")[0]
            data_item = {
                "start_time_s": float(r.start_time_s),
                "end_time_s": float(r.end_time_s),
                "description": str(r.description)
            }

            if last_end is not None and abs(data_item["start_time_s"] - last_end) > 1e-3:
                # print(video_id, segs[video_id])
                should_save = False
                break  # discontinuity in segments, skip this video
            last_end = data_item["end_time_s"]
            tmp_vid_segs.append(data_item)
        if should_save:
            segs.setdefault(video_id, []).extend(tmp_vid_segs)
    for vid in segs:
        segs[vid].sort(key=lambda x: x["start_time_s"])
    return segs

# ------------------------------- HRL-F (full) -----------------------------------
def evaluate_hrl_f(model: str, segments: Dict[str, List[Dict[str, float]]]) -> List[Tuple[str, str, str]]:
    items = []
    for vid, segs in segments.items():
        raw_video_path = stimuli_data_dir / (vid + ".mp4")
        video_path = f"file://{raw_video_path}"
        video_start_time = segs[0]["start_time_s"]
        previous_reactions = ""
        results = []
        for i, seg in enumerate(segs):
            start_time = seg["start_time_s"]
            end_time = seg["end_time_s"]
            # prepare video clip path by clipping video_path video with video_start_time and end_time
            # and save to tmp file

            clip_start = max(start_time - video_start_time, 0.0)
            clip_duration = max(end_time - start_time, 0.0)
            video_clip = raw_video_path
            video_clip_path = video_path
            if clip_duration > 0:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp_path = Path(tmp.name).resolve()
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{clip_start:.3f}",
                    "-i",
                    str(raw_video_path),
                    "-t",
                    f"{clip_duration:.3f}",
                    "-c",
                    "copy",
                    str(tmp_path),
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    tmp_path.unlink(missing_ok=True)
                else:
                    video_clip = tmp_path
                    video_clip_path = f"file://{tmp_path}"

            # prepare prompt
            current_time_window = f"{start_time:.2f}-{end_time:.2f}s"
            prompt = PROMPT_HRL_F.format(previous_reactions, current_time_window)
            predicted_reaction = generate_response(
                text=prompt,
                model=model,
                video_path=video_clip_path
            )
            
            # calculate similarity
            similarity = reaction_similarity(predicted_reaction, seg["description"])
            
            # save data
            results.append({
                "video_id": vid,
                "time_window": current_time_window,
                "predicted_reaction": predicted_reaction,
                "ground_truth_reaction": seg["description"],
                "similarity": similarity
            })
            
            # update previous reactions
            cur_reaction = f"Reaction for segment {i+1} ({current_time_window}): {seg['description']}"
            previous_reactions += "\n" + cur_reaction if previous_reactions else cur_reaction
    return items



# ------------------------------------ I/O --------------------------------------
def save_scores(path: Path, rows: List[Tuple[str, float]], header=("video_id", "score")):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

# ----------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=csv_data_dir)
    ap.add_argument("--out_f_full", default="scores_hrl_f.csv")
    ap.add_argument("--out_f_stream", default="scores_hrl_s.csv")
    args = ap.parse_args()

    segs = load_video_segments(Path(args.csv_dir))
    print(f"Loaded {len(segs)} videos from {args.csv_dir}")
    print("Sameple video segments:", list(segs.items())[:1])
    # scores_f = evaluate_hrl_f(segs, args.horizon_sec, args.batch_size)
    # save_scores(Path(args.out_f_full), scores_f, header=("video_id", "score_f"))

    # scores_s = evaluate_hrl_s(segs, args.batch_size, last_only=args.stream_last_only)
    # save_scores(Path(args.out_f_stream), scores_s, header=("video_id", "score_s"))

if __name__ == "__main__":
    csv_data_dir = Path("/orcd/scratch/seedfund/001/multimodal/qua/reaction_data/description")
    stimuli_data_dir = Path("/orcd/scratch/seedfund/001/multimodal/qua/reaction_data/stimuli")
    
    main()

