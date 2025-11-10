from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

DATA_ROOT = Path("/home/yibozhao83/Code/reaction_video_benchmark/benchmark")
SCRATCH_BENCHMARK_ROOT = Path("/home/yibozhao83/scratch/yibo/clean_human_reaction_data/benchmark")
VIDEO_ROOT = Path("/home/yibozhao83/scratch/yibo/clean_human_reaction_data")
OUTPUT_ROOT = Path("outputs")
CLIP_STORAGE_ROOT = DATA_ROOT / "tmp_clips"
QUESTION_STORAGE_ROOT = DATA_ROOT / "question_sets"


@dataclass(frozen=True)
class TaskSetting:
    key: str
    description: str
    default_prompt: str
    default_output_subdir: Path
    default_data_path: Optional[Path] = None
    default_question_path: Optional[Path] = None


TASK_SETTINGS: Dict[str, TaskSetting] = {
    "linking": TaskSetting(
        key="linking",
        description="Task 1: match video stimuli with the correct reaction segment.",
        default_prompt=(
            "You will receive a short video segment depicting a human reaction. "
            "Decide which reaction label best matches what the viewer expresses. "
            "Return only the most likely reaction name."
        ),
        default_output_subdir=OUTPUT_ROOT / "linking_benchmark",
        default_data_path=DATA_ROOT / "linking_task.jsonl",
    ),
    "highlight": TaskSetting(
        key="highlight",
        description="Task 2: quantify which clip provokes the strongest reaction.",
        default_prompt=(
            "You will watch a short human reaction clip. Rate how likely this specific clip is to trigger a strong "
            "emotional reaction in viewers on a scale of 1 (very mild) to 10 (extremely intense). "
            "Reply with a single number only."
        ),
        default_output_subdir=OUTPUT_ROOT / "reaction_benchmark",
        default_data_path=SCRATCH_BENCHMARK_ROOT / "gemini_annotated_intense_processed.jsonl",
        default_question_path=QUESTION_STORAGE_ROOT / "highlight_questions_sample_1000.jsonl",
    ),
    "reasoning": TaskSetting(
        key="reasoning",
        description="Task 3: explain why a reaction occurs based on cues in the clip.",
        default_prompt=(
            "After watching the clip and reading the viewer reaction, explain in 2 sentences which cues or events "
            "most likely caused that reaction. Reference visual or audio evidence explicitly."
        ),
        default_output_subdir=OUTPUT_ROOT / "reasoning_benchmark",
        default_data_path=DATA_ROOT / "reasoning_task.jsonl",
    ),
}

HIGHLIGHT_FIELDS = ("highlights_methodD",)

DEFAULT_TOTAL = 1000
DEFAULT_CHOICES = 4
DEFAULT_SEED = 13
DEFAULT_INTENSE_THRESHOLD = 7.0
DEFAULT_SCORE_MIN = 1.0
DEFAULT_SCORE_MAX = 10.0
