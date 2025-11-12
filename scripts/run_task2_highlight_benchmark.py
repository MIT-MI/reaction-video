#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.backbones import (  # noqa: E402
    BackboneConfig,
    GeminiBackbone,
    InternVideoBackbone,
    OpenAIBackbone,
    VLLMBackbone,
    VideoQueryBackbone,
)

from scripts.benchmark_settings import (  # noqa: E402
    CLIP_STORAGE_ROOT,
    DEFAULT_CHOICES,
    DEFAULT_INTENSE_THRESHOLD,
    DEFAULT_SCORE_MAX,
    DEFAULT_SCORE_MIN,
    DEFAULT_SEED,
    DEFAULT_TOTAL,
    HIGHLIGHT_FIELDS,
    TASK_SETTINGS,
    VIDEO_ROOT,
)

TASK_KEY = "highlight"
TASK_CONFIG = TASK_SETTINGS[TASK_KEY]
if TASK_CONFIG.default_data_path is None:
    raise ValueError("Highlight task requires a default data file, but none was configured.")
if TASK_CONFIG.default_question_path is None:
    raise ValueError("Highlight task requires a default questions file path, but none was configured.")

DEFAULT_DATA_PATH = str(TASK_CONFIG.default_data_path)
DEFAULT_VIDEO_ROOT = str(VIDEO_ROOT)
DEFAULT_OUTPUT_DIR = str(TASK_CONFIG.default_output_subdir)
DEFAULT_PROMPT = TASK_CONFIG.default_prompt
DEFAULT_QUESTIONS_PATH = str(TASK_CONFIG.default_question_path)


@dataclass
class SegmentSpec:
    video_id: str
    stimuli_path: str
    video_path: Path
    start: float
    end: float
    duration: float
    description: str
    label: str  # "intense" or "mild"
    source: str

    def clip_key(self) -> str:
        return f"{self.video_id}_{self.start:.3f}_{self.end:.3f}"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "clip_key": self.clip_key(),
            "video_id": self.video_id,
            "stimuli_path": self.stimuli_path,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "description": self.description,
            "label": self.label,
            "source": self.source,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], video_root: Path) -> "SegmentSpec":
        stimuli_path = payload["stimuli_path"]
        video_path = (video_root / stimuli_path).resolve()
        return cls(
            video_id=str(payload["video_id"]),
            stimuli_path=stimuli_path,
            video_path=video_path,
            start=float(payload["start"]),
            end=float(payload["end"]),
            duration=float(payload.get("duration", max(float(payload["end"]) - float(payload["start"]), 0.01))),
            description=payload.get("description", ""),
            label=str(payload.get("label", "mild")),
            source=str(payload.get("source", "unknown")),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2 (Quantifying) highlight benchmark.")
    parser.add_argument("--mode", choices=("generate", "evaluate"), required=True, help="Whether to generate questions or evaluate a model.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to the annotated JSONL file (used in generate mode).")
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT, help="Directory containing full-length stimuli videos.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to store clips, responses, and metrics.")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL, help="Number of questions to generate/evaluate (set <0 for max/all available).")
    parser.add_argument("--choices", type=int, default=DEFAULT_CHOICES, help="Number of choices per question (must be >=2).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")
    parser.add_argument("--intense-threshold", type=float, default=DEFAULT_INTENSE_THRESHOLD, help="Trigger score cutoff (<= threshold is mild).")
    parser.add_argument("--prompt", default=None, help="Inline prompt text. Overrides --prompt-file when provided.")
    parser.add_argument("--prompt-file", default=None, help="Optional file containing the scoring prompt.")
    parser.add_argument("--backbone", choices=("vllm", "gemini", "openai", "internvideo", "minicpm"), required=False, help="MLLM backbone to use.")
    parser.add_argument("--model-name", default=None, help="Model identifier shared by all backbones.")
    parser.add_argument("--api-key", default=None, help="API key for OpenAI/Gemini (optional for local vLLM).")
    parser.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible endpoints (vLLM or Azure).")
    parser.add_argument("--request-timeout", type=int, default=120, help="Request timeout (seconds).")
    parser.add_argument("--responses-path", default=None, help="Override path for raw responses JSONL.")
    parser.add_argument("--scores-path", default=None, help="Override path for scores CSV.")
    parser.add_argument("--metrics-path", default=None, help="Override path for metrics JSON.")
    parser.add_argument("--clip-dir", default=None, help="Directory to cache extracted clips (defaults to benchmark storage).")
    parser.add_argument("--questions-path", default=DEFAULT_QUESTIONS_PATH, help="Location to save or load question sets (JSONL).")
    parser.add_argument("--flush-num", type=int, default=100, help="Number of freshly generated questions to buffer before flushing to disk.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for scoring.")
    parser.add_argument("--score-min", type=float, default=DEFAULT_SCORE_MIN, help="Lower bound for model scores.")
    parser.add_argument("--score-max", type=float, default=DEFAULT_SCORE_MAX, help="Upper bound for model scores.")
    parser.add_argument("--dry-run", action="store_true", help="Skip model calls and generate heuristic scores.")
    parser.add_argument("--input-mode", choices=("video", "image"), default="video", help="Input mode: 'video' for native video input, 'image' for frame extraction.")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample when using --input-mode=image.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.flush_num < 1:
        parser.error("--flush-num must be >= 1")
    return args


def load_prompt(args: argparse.Namespace, fallback_prompt: str) -> str:
    if args.prompt:
        return args.prompt.strip()
    if args.prompt_file:
        path = Path(args.prompt_file)
        return path.read_text().strip()
    return fallback_prompt


def deterministic_sim_seed(question_id: int, choice_id: int, clip_key: str) -> int:
    payload = f"{question_id}:{choice_id}:{clip_key}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


def load_segments(data_path: Path, video_root: Path, intense_threshold: float) -> Tuple[List[SegmentSpec], List[SegmentSpec]]:
    intense_segments: List[SegmentSpec] = []
    mild_segments: List[SegmentSpec] = []

    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            stimuli_rel = record.get("stimuli_path")
            if not stimuli_rel:
                continue
            full_video_path = (video_root / stimuli_rel).resolve()
            if not full_video_path.exists():
                logging.warning("Stimuli video missing: %s", full_video_path)
                continue

            for segment in extract_highlight_segments(record, stimuli_rel, full_video_path, intense_threshold):
                if segment.label == "intense":
                    intense_segments.append(segment)
                else:
                    mild_segments.append(segment)

            # Fallback: treat generic segments as mild for additional negatives.
            for seg in record.get("segments", []):
                start = float(seg.get("start_time", 0))
                end = float(seg.get("end_time", start))
                duration = max(end - start, 0.01)
                desc = seg.get("description", "Generic segment")
                mild_segments.append(
                    SegmentSpec(
                        video_id=str(record.get("video_id")),
                        stimuli_path=stimuli_rel,
                        video_path=full_video_path,
                        start=start,
                        end=end,
                        duration=duration,
                        description=desc,
                        label="mild",
                        source="segments",
                    )
                )

    return intense_segments, mild_segments


def extract_highlight_segments(record: Dict[str, Any], stimuli_path: str, video_path: Path, intense_threshold: float) -> List[SegmentSpec]:
    segments: List[SegmentSpec] = []
    for field in HIGHLIGHT_FIELDS:
        for entry in record.get(field, []) or []:
            aux = entry.get("aux") or {}
            label = infer_label(field, aux, intense_threshold)
            if not label:
                continue
            for seg in entry.get("highlight_segments", []):
                start = float(seg.get("start", seg.get("start_time", 0.0)))
                end = float(seg.get("end", seg.get("end_time", start)))
                duration = seg.get("duration")
                if duration is None:
                    duration = max(end - start, 0.01)
                if end <= start:
                    continue
                description = entry.get("description") or aux.get("rationale") or "highlight"
                segments.append(
                    SegmentSpec(
                        video_id=str(record.get("video_id")),
                        stimuli_path=stimuli_path,
                        video_path=video_path,
                        start=start,
                        end=end,
                        duration=float(duration),
                        description=description,
                        label=label,
                        source=field,
                    )
                )
    return segments


def infer_label(field: str, aux: Dict[str, Any], intense_threshold: float) -> Optional[str]:
    direct_label = aux.get("intensity_label") or aux.get("raw_intensity_label")
    if direct_label:
        direct_label = direct_label.lower()
        if direct_label in {"intense", "mild"}:
            return direct_label

    trigger_score = aux.get("trigger_score_1_10")
    if isinstance(trigger_score, (int, float)):
        return "intense" if trigger_score >= intense_threshold else "mild"

    if field in {"highlights_methodB", "highlights_methodD"}:
        return "intense"
    return None


def cap_score(value: Optional[float], min_score: float, max_score: float) -> Optional[float]:
    if value is None:
        return None
    return max(min_score, min(max_score, value))


def ensure_clip(segment: SegmentSpec, clip_dir: Path) -> Path:
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / f"{segment.clip_key()}.mp4"
    if clip_path.exists():
        return clip_path

    logging.info("Extracting video clip: %s [%.3fs-%.3fs]", segment.video_id, segment.start, segment.end)
    duration = max(segment.end - segment.start, 0.05)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{segment.start:.3f}",
        "-i",
        str(segment.video_path),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(clip_path),
    ]
    subprocess.run(cmd, check=True)
    return clip_path


def load_question_set(
    question_path: Path, video_root: Path, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    if not question_path.exists():
        raise FileNotFoundError(f"Question set not found: {question_path}")

    questions: List[Dict[str, Any]] = []
    with question_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            options = []
            for payload in record.get("options", []):
                segment = SegmentSpec.from_payload(payload, video_root)
                options.append({"segment": segment, "score": None, "raw_response": None, "clip_path": None})
            question_id = int(record.get("question_id", len(questions) + 1))
            questions.append({"question_id": question_id, "options": options})
            if limit is not None and len(questions) >= limit:
                break
    return questions


def load_existing_question_metadata(question_path: Path) -> Tuple[List[Set[str]], int, int]:
    if not question_path.exists():
        return [], 0, 0

    clip_sets: List[Set[str]] = []
    count = 0
    max_qid = 0
    with question_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            clip_keys = {opt.get("clip_key") for opt in record.get("options", []) if opt.get("clip_key")}
            clip_sets.append(clip_keys)
            count += 1
            qid = int(record.get("question_id", count))
            if qid > max_qid:
                max_qid = qid
    return clip_sets, count, max_qid


def simulate_score(segment: SegmentSpec, seed_value: int) -> Tuple[float, str]:
    rng = random.Random(seed_value)
    base = 8.0 if segment.label == "intense" else 3.0
    noise = rng.uniform(-0.5, 0.5)
    score = round(base + noise, 3)
    return score, f"{score}"


def build_backbone(args: argparse.Namespace) -> Optional[VideoQueryBackbone]:
    if args.dry_run:
        return None
    if not args.backbone:
        raise ValueError("Please specify --backbone when not running in --dry-run mode.")
    if not args.model_name:
        raise ValueError("Please provide --model-name for model-backed runs.")

    config = BackboneConfig(
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        request_timeout=args.request_timeout,
        mode=args.input_mode,
        num_frames=args.num_frames,
    )

    if args.backbone == "vllm":
        return VLLMBackbone(config)
    if args.backbone == "openai":
        return OpenAIBackbone(config)
    if args.backbone == "gemini":
        return GeminiBackbone(config)
    if args.backbone == "internvideo":
        return InternVideoBackbone(config)
    if args.backbone == "minicpm":
        from eval.backbones import MiniCPMBackbone
        return MiniCPMBackbone(config)
    if args.backbone == "videochat" or args.backbone == "videochat-flash":
        from eval.backbones import VideoChatFlashBackbone
        return VideoChatFlashBackbone(config)
    raise ValueError(f"Unsupported backbone: {args.backbone}")


def compute_metrics(questions: Sequence[Dict[str, Any]], total_requested: int) -> Dict[str, Any]:
    top1_hits = 0
    top3_hits = 0
    eligible_questions = 0
    total_clips = sum(len(question["options"]) for question in questions)
    scored_clips = sum(1 for question in questions for option in question["options"] if option["score"] is not None)
    intense_scores: List[float] = []
    mild_scores: List[float] = []

    for question in questions:
        options = question["options"]
        if any(opt["score"] is None for opt in options):
            continue
        eligible_questions += 1
        sorted_options = sorted(options, key=lambda item: item["score"], reverse=True)
        intense_option = next((opt for opt in options if opt["segment"].label == "intense"), None)
        if intense_option is None:
            continue

        intense_scores.append(float(intense_option["score"]))
        mild_scores.extend(float(opt["score"]) for opt in options if opt["segment"].label == "mild")

        if sorted_options[0] is intense_option:
            top1_hits += 1

        top3_subset = sorted_options[: min(3, len(sorted_options))]
        if intense_option in top3_subset:
            top3_hits += 1

    metrics = {
        "questions_requested": total_requested,
        "questions_evaluated": len(questions),
        "questions_with_scores": eligible_questions,
        "top1_accuracy": (top1_hits / eligible_questions) if eligible_questions else 0.0,
        "top3_accuracy": (top3_hits / eligible_questions) if eligible_questions else 0.0,
        "mean_intense_score": (sum(intense_scores) / len(intense_scores)) if intense_scores else None,
        "mean_mild_score": (sum(mild_scores) / len(mild_scores)) if mild_scores else None,
        "score_gap": (
            (sum(intense_scores) / len(intense_scores)) - (sum(mild_scores) / len(mild_scores))
        )
        if intense_scores and mild_scores
        else None,
        "clip_coverage": scored_clips / total_clips if total_clips else 0.0,
    }
    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    video_root = Path(args.video_root)
    if not video_root.exists():
        logging.error("Video root not found: %s", video_root)
        sys.exit(1)

    if args.mode == "generate":
        run_generate(args, video_root)
        return

    run_evaluate(args, video_root)


def run_generate(args: argparse.Namespace, video_root: Path) -> None:
    data_path = Path(args.data_path)
    if not data_path.exists():
        logging.error("Data file not found: %s", data_path)
        sys.exit(1)

    rng = random.Random(args.seed)
    logging.info("Loading candidate segments from %s", data_path)
    intense_segments, mild_segments = load_segments(data_path, video_root, args.intense_threshold)
    logging.info("Found %d intense and %d mild segments before sampling.", len(intense_segments), len(mild_segments))

    if args.choices < 2:
        logging.error("--choices must be >= 2")
        sys.exit(1)
    if len(mild_segments) < args.choices - 1:
        logging.error("Not enough mild segments (%d) to form a single question with %d choices.", len(mild_segments), args.choices)
        sys.exit(1)

    question_path = Path(args.questions_path)
    existing_clip_sets, existing_count, max_qid = load_existing_question_metadata(question_path)
    desired_total = args.total if args.total > 0 else len(intense_segments)
    desired_total = min(desired_total, len(intense_segments))
    if desired_total <= 0:
        desired_total = len(intense_segments)

    if existing_count >= desired_total:
        logging.info(
            "Question set already has %d entries (>= desired %d). Skipping generation.",
            existing_count,
            desired_total,
        )
        return

    question_path.parent.mkdir(parents=True, exist_ok=True)

    rng.shuffle(intense_segments)
    rng.shuffle(mild_segments)

    overlap_limit = 1
    highlight_idx = 0
    produced_new = 0
    buffer: List[Dict[str, Any]] = []
    next_question_id = max_qid
    total_needed = desired_total - existing_count if desired_total else len(intense_segments)

    eligible_mild_pool = mild_segments[:]  # reuse for sampling

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        mode = "a"
        with question_path.open(mode, encoding="utf-8") as handle:
            for record in buffer:
                handle.write(json.dumps(record) + "\n")
        buffer = []

    try:
        while (total_needed is None or produced_new < total_needed) and highlight_idx < len(intense_segments):
            highlight = intense_segments[highlight_idx]
            highlight_idx += 1

            eligible_mild = [seg for seg in eligible_mild_pool if seg.clip_key() != highlight.clip_key()]
            if len(eligible_mild) < args.choices - 1:
                logging.debug("Skipping highlight %s due to insufficient distinct mild segments.", highlight.clip_key())
                continue

            success = False
            for _ in range(25):
                mild_sample = rng.sample(eligible_mild, k=args.choices - 1)
                options = mild_sample + [highlight]
                clip_keys = {seg.clip_key() for seg in options}

                if all(len(clip_keys & prev_keys) <= overlap_limit for prev_keys in existing_clip_sets):
                    rng.shuffle(options)
                    next_question_id += 1
                    record = {
                        "question_id": next_question_id,
                        "options": [seg.to_payload() for seg in options],
                    }
                    existing_clip_sets.append(clip_keys)
                    buffer.append(record)
                    produced_new += 1
                    success = True
                    current_total = existing_count + produced_new
                    sys.stdout.write(f"\rGenerated {current_total} questions...")
                    sys.stdout.flush()
                    if len(buffer) >= args.flush_num:
                        flush_buffer()
                    break

            if not success:
                logging.debug("Could not form a unique question for highlight %s after retries.", highlight.clip_key())

            if total_needed is not None and produced_new >= total_needed:
                break
    finally:
        flush_buffer()
        if produced_new:
            sys.stdout.write("\n")
            sys.stdout.flush()

    final_total = existing_count + produced_new
    if produced_new == 0:
        logging.warning("No new questions were generated. Consider adjusting thresholds or dataset size.")
    elif final_total < desired_total:
        logging.warning(
            "Generated %d new questions (total=%d), fewer than requested %d due to uniqueness constraints.",
            produced_new,
            final_total,
            desired_total,
        )
    else:
        logging.info("Generated %d new questions; question set now has %d entries.", produced_new, final_total)


def load_existing_results(responses_path: Path, scores_path: Path) -> Set[Tuple[int, int]]:
    """Load already-processed (question_id, choice_id) pairs from existing output files."""
    completed: Set[Tuple[int, int]] = set()
    
    if responses_path.exists():
        try:
            with responses_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    q_id = record.get("question_id")
                    c_id = record.get("choice_id")
                    if q_id is not None and c_id is not None:
                        completed.add((int(q_id), int(c_id)))
        except (json.JSONDecodeError, IOError) as e:
            logging.warning("Error reading existing responses from %s: %s", responses_path, e)
    
    if scores_path.exists():
        try:
            with scores_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    q_id = row.get("question_id")
                    c_id = row.get("choice_id")
                    if q_id and c_id:
                        completed.add((int(q_id), int(c_id)))
        except (csv.Error, IOError) as e:
            logging.warning("Error reading existing scores from %s: %s", scores_path, e)
    
    return completed


def run_evaluate(args: argparse.Namespace, video_root: Path) -> None:
    question_path = Path(args.questions_path)
    limit = args.total if args.total > 0 else None
    questions = load_question_set(question_path, video_root, limit)
    if not questions:
        logging.error("Question set is empty. Provide a valid questions file.")
        sys.exit(1)
    logging.info("Loaded %d questions from %s", len(questions), question_path)
    if args.total > 0 and len(questions) < args.total:
        logging.warning(
            "Requested to evaluate %d questions but only %d are available in the question set.",
            args.total,
            len(questions),
        )

    prompt = load_prompt(args, DEFAULT_PROMPT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = Path(args.clip_dir) if args.clip_dir else CLIP_STORAGE_ROOT
    responses_path = Path(args.responses_path) if args.responses_path else output_dir / "responses.jsonl"
    scores_path = Path(args.scores_path) if args.scores_path else output_dir / "scores.csv"
    metrics_path = Path(args.metrics_path) if args.metrics_path else output_dir / "metrics.json"

    backbone = build_backbone(args)

    responses_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results to skip already-completed jobs
    completed_jobs = load_existing_results(responses_path, scores_path)
    if completed_jobs:
        logging.info("Found %d already-completed cases to skip.", len(completed_jobs))

    for question in questions:
        for idx, option in enumerate(question["options"], start=1):
            clip_key = option["segment"].clip_key()
            option["sim_seed"] = deterministic_sim_seed(question["question_id"], idx, clip_key)

    # Initialize CSV file with header if it doesn't exist
    if not scores_path.exists():
        with scores_path.open("w", newline="", encoding="utf-8") as score_file:
            score_writer = csv.DictWriter(
                score_file,
                fieldnames=["question_id", "choice_id", "video_id", "clip_path", "label", "score"],
            )
            score_writer.writeheader()

    # Thread-safe locks for incremental file writes
    response_lock = threading.Lock()
    score_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed_count = [0]  # Use list for mutable counter

    def process_option(question_id: int, choice_id: int, option: Dict[str, Any]) -> Tuple[int, int, Dict[str, Any], Dict[str, Any]]:
        segment: SegmentSpec = option["segment"]
        clip_path = ensure_clip(segment, clip_dir)
        clip_path_str = str(clip_path)
        try:
            if args.dry_run:
                score, raw_response = simulate_score(segment, option["sim_seed"])
            else:
                if backbone is None:
                    raise RuntimeError("Backbone is not initialized.")
                # import pdb; pdb.set_trace()
                score, raw_response = backbone.score_clip(clip_path, prompt)  # type: ignore[arg-type]
        finally:
            try:
                clip_path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass

        option["clip_path"] = clip_path_str
        option["score"] = cap_score(score, args.score_min, args.score_max)
        option["raw_response"] = raw_response

        response_record = {
            "question_id": question_id,
            "choice_id": choice_id,
            "video_id": segment.video_id,
            "clip_path": clip_path_str,
            "label": segment.label,
            "description": segment.description,
            "source": segment.source,
            "start": segment.start,
            "end": segment.end,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_score": option["score"],
        }

        score_record = {
            "question_id": question_id,
            "choice_id": choice_id,
            "video_id": segment.video_id,
            "clip_path": clip_path_str,
            "label": segment.label,
            "score": option["score"],
        }

        # Write results immediately (thread-safe)
        with response_lock:
            with responses_path.open("a", encoding="utf-8") as response_file:
                response_file.write(json.dumps(response_record) + "\n")

        with score_lock:
            with scores_path.open("a", newline="", encoding="utf-8") as score_file:
                score_writer = csv.DictWriter(
                    score_file,
                    fieldnames=["question_id", "choice_id", "video_id", "clip_path", "label", "score"],
                )
                score_writer.writerow(score_record)

        with progress_lock:
            completed_count[0] += 1
            logging.info(
                "Completed %d/%d: Q%d C%d [%s] | Ground truth: %s | Response: %s | Score: %.2f",
                completed_count[0],
                len(jobs),
                question_id,
                choice_id,
                segment.video_id,
                segment.label,
                raw_response,
                option["score"] if option["score"] is not None else -1,
            )

        return question_id, choice_id, response_record, score_record

    # Build job list, excluding already-completed cases
    jobs: List[Tuple[int, int, Dict[str, Any]]] = []
    for question in questions:
        for choice_idx, option in enumerate(question["options"], start=1):
            job_key = (question["question_id"], choice_idx)
            if job_key not in completed_jobs:
                jobs.append((question["question_id"], choice_idx, option))

    if not jobs:
        logging.info("All cases already completed. Skipping evaluation.")
    else:
        logging.info("Processing %d remaining cases (out of %d total).", len(jobs), len(jobs) + len(completed_jobs))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_option, q_id, choice_id, option) for q_id, choice_id, option in jobs]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error("Error processing option: %s", e, exc_info=True)

    # Best-effort cleanup of transient clip directory when empty.
    try:
        if clip_dir.exists() and not any(clip_dir.iterdir()):
            clip_dir.rmdir()
    except OSError:
        pass

    # Reload all scores from file to update questions structure for metrics computation
    logging.info("Loading all results to compute metrics...")
    if scores_path.exists():
        score_map: Dict[Tuple[int, int], float] = {}
        with scores_path.open("r", encoding="utf-8") as score_file:
            reader = csv.DictReader(score_file)
            for row in reader:
                q_id = int(row["question_id"])
                c_id = int(row["choice_id"])
                score = row.get("score")
                if score and score.strip():
                    score_map[(q_id, c_id)] = float(score)
        
        # Update questions with loaded scores
        for question in questions:
            for choice_idx, option in enumerate(question["options"], start=1):
                job_key = (question["question_id"], choice_idx)
                if job_key in score_map:
                    option["score"] = score_map[job_key]

    requested_total = len(questions) if args.total < 0 else args.total
    metrics = compute_metrics(questions, requested_total)
    if args.model_name:
        metrics["model_name"] = args.model_name
    if args.backbone:
        metrics["backbone"] = args.backbone
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logging.info("Responses saved to %s", responses_path)
    logging.info("Scores saved to %s", scores_path)
    logging.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
