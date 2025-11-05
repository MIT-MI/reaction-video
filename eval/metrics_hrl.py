PROMPT_HRL_F = """A viewer is watching a video segment from {start_time_v} to {end_time_v}.

Previously, a viewer watched the video from {start_time_r} to {end_time_r} and reacted as follows:
{previous_reactions}

Now, based on how the viewer reacted previously, predict the viewer’s next reaction for {current_time_window}, considering both the video and the previous reactions.

Write the reaction as exactly one concise English sentence describing the visible facial expression or emotion.

Predicted reaction:"""




# eval_hrl_min.py
from pathlib import Path
from typing import List, Tuple, Dict, Set
import pandas as pd
from tqdm import tqdm
import argparse
import json
import math
import subprocess
import tempfile
import os
import requests
from collections import Counter
from models import generate_response

API_URL = "http://127.0.0.1:8080/similarity"

# ---------------------- similarity metric (single function) ----------------------
def reaction_similarity(a: str, b: str) -> float:
    """
    Compute semantic similarity between two texts using the running FastAPI service.
    Returns a float in [-1, 1].
    """
    try:
        response = requests.post(
            API_URL,
            json={"s1": a, "s2": b},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        return float(result.get("similarity", 0.0))
    except Exception as e:
        print(f"[Error] Failed to get similarity: {e}")
        return 0.0

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

def load_evaluated_videos(output_path: Path) -> Dict[str, List[Dict]]:
    """Load already evaluated video results from existing JSON file."""
    evaluated = {}
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                evaluated = json.load(f)
                print(f"Found {len(evaluated)} already evaluated videos in {output_path}")
        except Exception as e:
            print(f"Warning: Could not load existing results from {output_path}: {e}")
    return evaluated

def save_detailed_results(output_path: Path, results: Dict[str, List[Dict]]):
    """Save detailed evaluation results to JSON as a dictionary with video IDs as keys."""
    if not results:
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# ------------------------------- HRL-F (full) -----------------------------------
def evaluate_hrl(
    model: str, 
    eval_mode: str,
    segments: Dict[str, List[Dict[str, float]]], 
    output_path: Path,
    stimuli_dir: Path,
    save_interval: int = 10
) -> Dict[str, List[Dict]]:
    """
    Evaluate HRL-F metric with incremental saving and resumption support.
    
    Args:
        model: Model name for inference
        segments: Dictionary mapping video IDs to their segments
        output_path: Path to save results JSON
        save_interval: Save results every N videos (default: 10)
    
    Returns:
        Dictionary mapping video IDs to their evaluation results
    """
    # Load already evaluated videos to support resumption
    all_results = load_evaluated_videos(output_path)
    
    pending_results = {}
    video_count = 0
    
    # Filter out already evaluated videos
    videos_to_process = [(vid, segs) for vid, segs in segments.items() if vid not in all_results]
    
    if all_results:
        print(f"Skipping {len(all_results)} already evaluated videos")
    print(f"Processing {len(videos_to_process)} videos")
    
    for vid, segs in tqdm(videos_to_process, desc="Evaluating HRL-F"):
        raw_video_path = stimuli_dir / (vid + ".mp4")
        
        # Check if video file exists
        if not raw_video_path.exists():
            print(f"Warning: Video file not found: {raw_video_path}")
            continue
        
        video_start_time = segs[0]["start_time_s"]
        previous_reactions = ""
        video_results = []
        
        for i, seg in enumerate(segs):
            start_time = seg["start_time_s"]
            end_time = seg["end_time_s"]
            tmp_video_path = None
            
            try:
                # Create temporary file for video clip
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_video_path_full = tmp.name
                
                # Calculate duration from video_start_time to end_time
                duration = end_time - video_start_time
                
                # Use ffmpeg to clip the video
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", str(video_start_time),
                    "-i", str(raw_video_path),
                    "-t", str(duration),
                    "-c", "copy",
                    tmp_video_path_full
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                
                video_clip_path = f"file://{tmp_video_path_full}"

                # Prepare prompt
                current_time_window = f"{start_time:.2f}-{end_time:.2f}s"
                prompt = PROMPT_HRL_F.format(
                    start_time_v=video_start_time, 
                    end_time_v=end_time, 
                    start_time_r=video_start_time, 
                    end_time_r=start_time, 
                    previous_reactions=previous_reactions,
                    current_time_window=current_time_window
                )
                
                # import pdb; pdb.set_trace()
                
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
                    "segment_index": i,
                    "time_window": current_time_window,
                    "predicted_reaction": predicted_reaction,
                    "ground_truth_reaction": seg["description"],
                    "similarity": similarity
                }
                video_results.append(result)
                
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
        
        # Store video results
        if video_results:
            all_results[vid] = video_results
            pending_results[vid] = video_results
        
        # Increment video counter
        video_count += 1
        
        # Save results incrementally every save_interval videos
        if video_count % save_interval == 0 and pending_results:
            save_detailed_results(output_path, all_results)
            total_segments = sum(len(v) for v in pending_results.values())
            print(f"Saved results for {video_count} videos ({total_segments} segments)")
            pending_results = {}
    
    # Save any remaining results
    if pending_results:
        save_detailed_results(output_path, all_results)
        total_segments = sum(len(v) for v in pending_results.values())
        print(f"Saved final batch: {len(pending_results)} videos ({total_segments} segments)")
    
    return all_results



if __name__ == "__main__":
    csv_data_dir = Path("/orcd/scratch/seedfund/001/multimodal/qua/reaction_data/description")
    stimuli_data_dir = Path("/orcd/scratch/seedfund/001/multimodal/qua/reaction_data/stimuli")
    result_dir = Path("orcd/home/002/qua/code/reaction/reaction-video/results")

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=str(csv_data_dir), help="Directory containing CSV segment descriptions")
    ap.add_argument("--stimuli_dir", default=str(stimuli_data_dir), help="Directory containing stimuli videos")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name for evaluation")
    ap.add_argument("--out_f_full", default=str(result_dir / "results_hrl_f.json"), help="Output JSON for HRL-F results")
    ap.add_argument("--save_interval", type=int, default=10, help="Save results every N videos")
    ap.add_argument("--eval_mode", default="lvh", help="Evaluation mode: l for language, v for vision, h for hybrid")
    
    # ap.add_argument("--")
    args = ap.parse_args()

    # Load video segments
    segs = load_video_segments(Path(args.csv_dir))
    print(f"Loaded {len(segs)} videos from {args.csv_dir}")
    
    # Evaluate HRL-F with incremental saving and resumption support
    print(f"\nEvaluating HRL-F with model: {args.model}")
    print(f"Results will be saved to: {args.out_f_full}")
    print(f"Saving every {args.save_interval} videos")
    

    if not Path(args.out_f_full).parent.exists():
        Path(args.out_f_full).parent.mkdir(parents=True, exist_ok=True)

    results_f = evaluate_hrl(
        model=args.model,
        eval_mode=args.eval_mode,
        segments=segs,
        output_path=Path(args.out_f_full),
        stimuli_dir=Path(args.stimuli_dir),
        save_interval=args.save_interval
    )
