PROMPT_HRL_F = """
Previous Info: {}

Predict the reaction
"""




# eval_hrl_min.py
from pathlib import Path
from typing import List, Tuple, Dict, Set
import pandas as pd
from tqdm import tqdm
import argparse
import csv
import math
import subprocess
import tempfile
import os
from collections import Counter
from models import generate_response

# ---------------------- similarity metric (single function) ----------------------
def reaction_similarity(a: str, b: str) -> float:
    
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

def load_evaluated_videos(output_path: Path) -> Set[str]:
    """Load the set of already evaluated video IDs from existing results file."""
    evaluated = set()
    if output_path.exists():
        try:
            df = pd.read_csv(output_path)
            if "video_id" in df.columns:
                evaluated = set(df["video_id"].unique())
                print(f"Found {len(evaluated)} already evaluated videos in {output_path}")
        except Exception as e:
            print(f"Warning: Could not load existing results from {output_path}: {e}")
    return evaluated

def save_detailed_results(output_path: Path, results: List[Dict], append: bool = False):
    """Save detailed evaluation results to CSV."""
    if not results:
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine fieldnames from first result
    fieldnames = list(results[0].keys())
    
    mode = "a" if append and output_path.exists() else "w"
    write_header = not (append and output_path.exists())
    
    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)

# ------------------------------- HRL-F (full) -----------------------------------
def evaluate_hrl_f(
    model: str, 
    segments: Dict[str, List[Dict[str, float]]], 
    output_path: Path,
    save_interval: int = 10
) -> List[Dict]:
    """
    Evaluate HRL-F metric with incremental saving and resumption support.
    
    Args:
        model: Model name for inference
        segments: Dictionary mapping video IDs to their segments
        output_path: Path to save results CSV
        save_interval: Save results every N videos (default: 10)
    
    Returns:
        List of all evaluation results
    """
    # Load already evaluated videos to support resumption
    evaluated_videos = load_evaluated_videos(output_path)
    
    all_results = []
    pending_results = []
    video_count = 0
    
    # Filter out already evaluated videos
    videos_to_process = [(vid, segs) for vid, segs in segments.items() if vid not in evaluated_videos]
    
    if evaluated_videos:
        print(f"Skipping {len(evaluated_videos)} already evaluated videos")
    print(f"Processing {len(videos_to_process)} videos")
    
    for vid, segs in tqdm(videos_to_process, desc="Evaluating HRL-F"):
        raw_video_path = stimuli_data_dir / (vid + ".mp4")
        
        # Check if video file exists
        if not raw_video_path.exists():
            print(f"Warning: Video file not found: {raw_video_path}")
            continue
        
        video_start_time = segs[0]["start_time_s"]
        previous_reactions = ""
        
        for i, seg in enumerate(segs):
            start_time = seg["start_time_s"]
            end_time = seg["end_time_s"]
            tmp_video_path = None
            
            try:
                # Create temporary file for video clip
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_video_path = tmp.name
                
                # Calculate duration from video_start_time to end_time
                duration = end_time - video_start_time
                
                # Use ffmpeg to clip the video
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", str(video_start_time),
                    "-i", str(raw_video_path),
                    "-t", str(duration),
                    "-c", "copy",
                    tmp_video_path
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                
                video_clip_path = f"file://{tmp_video_path}"

                # Prepare prompt
                current_time_window = f"{start_time:.2f}-{end_time:.2f}s"
                prompt = PROMPT_HRL_F.format(previous_reactions, current_time_window)
                
                # Generate prediction
                predicted_reaction = generate_response(
                    text=prompt,
                    model=model,
                    video_path=video_clip_path
                )
                
                # Calculate similarity
                similarity = reaction_similarity(predicted_reaction, seg["description"])
                
                # Save result
                result = {
                    "video_id": vid,
                    "segment_index": i,
                    "time_window": current_time_window,
                    "predicted_reaction": predicted_reaction,
                    "ground_truth_reaction": seg["description"],
                    "similarity": similarity
                }
                pending_results.append(result)
                all_results.append(result)
                
                # Update previous reactions for next iteration
                cur_reaction = f"Reaction for segment {i+1} ({current_time_window}): {seg['description']}"
                previous_reactions += "\n" + cur_reaction if previous_reactions else cur_reaction
                
            except Exception as e:
                print(f"Error processing {vid} segment {i}: {e}")
                continue
            
            finally:
                # Clean up temporary video file immediately
                if tmp_video_path and os.path.exists(tmp_video_path):
                    try:
                        os.unlink(tmp_video_path)
                    except OSError as e:
                        print(f"Warning: Could not delete temporary file {tmp_video_path}: {e}")
        
        # Increment video counter
        video_count += 1
        
        # Save results incrementally every save_interval videos
        if video_count % save_interval == 0 and pending_results:
            save_detailed_results(output_path, pending_results, append=True)
            print(f"Saved results for {video_count} videos ({len(pending_results)} segments)")
            pending_results = []
    
    # Save any remaining results
    if pending_results:
        save_detailed_results(output_path, pending_results, append=True)
        print(f"Saved final batch of {len(pending_results)} segments")
    
    return all_results



# ------------------------------------ I/O --------------------------------------
def save_scores(path: Path, rows: List[Tuple[str, float]], header=("video_id", "score")):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

# ----------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=str(csv_data_dir), help="Directory containing CSV segment descriptions")
    ap.add_argument("--model", default="qwen2-vl-7b-instruct", help="Model name for evaluation")
    ap.add_argument("--out_f_full", default="results_hrl_f.csv", help="Output CSV for HRL-F results")
    ap.add_argument("--save_interval", type=int, default=10, help="Save results every N videos")
    args = ap.parse_args()

    # Load video segments
    segs = load_video_segments(Path(args.csv_dir))
    print(f"Loaded {len(segs)} videos from {args.csv_dir}")
    print("Sample video segments:", list(segs.items())[:1])
    
    # Evaluate HRL-F with incremental saving and resumption support
    print(f"\nEvaluating HRL-F with model: {args.model}")
    print(f"Results will be saved to: {args.out_f_full}")
    print(f"Saving every {args.save_interval} videos")
    
    results_f = evaluate_hrl_f(
        model=args.model,
        segments=segs,
        output_path=Path(args.out_f_full),
        save_interval=args.save_interval
    )
    
    # Calculate and print statistics
    if results_f:
        avg_similarity = sum(r["similarity"] for r in results_f) / len(results_f)
        print(f"\nEvaluation complete!")
        print(f"Total segments evaluated: {len(results_f)}")
        print(f"Average similarity score: {avg_similarity:.4f}")
    else:
        print("\nNo results generated.")
    
    # TODO: Implement HRL-S (streaming) evaluation
    # scores_s = evaluate_hrl_s(segs, args.batch_size, last_only=args.stream_last_only)
    # save_scores(Path(args.out_f_stream), scores_s, header=("video_id", "score_s"))

if __name__ == "__main__":
    csv_data_dir = Path("/orcd/scratch/seedfund/001/multimodal/qua/reaction_data/description")
    stimuli_data_dir = Path("/orcd/scratch/seedfund/001/multimodal/qua/reaction_data/stimuli")
    
    main()

