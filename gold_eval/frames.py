"""Clip download (R2 public bucket) + 1 fps frame extraction, cached and deterministic.

Frames are the common input currency for every backbone (ADR-0003: same sampled-frame budget
for all models). Extraction: ffmpeg fps filter, max side 512, JPEG q4. If a clip yields more
than max_frames, frames are subsampled evenly (deterministic).
"""

from __future__ import annotations

import base64
import subprocess
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
CLIPS = DATA / "clips"
FRAMES = DATA / "frames"


def fetch_clip(r2_base: str, key: str) -> Path:
    dest = CLIPS / key
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    # R2's public dev endpoint 403s the default Python-urllib User-Agent.
    req = urllib.request.Request(r2_base.rstrip("/") + "/" + key,
                                 headers={"User-Agent": "Mozilla/5.0 (gold-eval)"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.rename(dest)
    return dest


def extract_frames(clip: Path, fps: float = 1.0, max_side: int = 512,
                   max_frames: int | None = None,
                   start_s: float | None = None, end_s: float | None = None) -> list[Path]:
    win = f"_w{start_s:.1f}-{end_s:.1f}" if start_s is not None else ""
    out_dir = FRAMES / f"{clip.parent.name}_{clip.stem}_fps{fps}_s{max_side}{win}"
    if not out_dir.is_dir() or not any(out_dir.glob("*.jpg")):
        tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        scale = f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})'"
        seek = ([] if start_s is None else ["-ss", f"{start_s}"]) + \
               ([] if end_s is None else ["-to", f"{end_s}"])
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *seek, "-i", str(clip),
                 "-vf", f"fps={fps},{scale}", "-q:v", "4", str(tmp_dir / "%05d.jpg")],
                check=True)
        except subprocess.CalledProcessError:
            pass
        if not any(tmp_dir.glob("*.jpg")):
            # degenerate clip/window (shorter than 1/fps): fall back to the first frame
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *seek, "-i", str(clip),
                 "-vf", scale, "-frames:v", "1", "-q:v", "4", str(tmp_dir / "00001.jpg")],
                check=True)
        tmp_dir.rename(out_dir)
    frames = sorted(out_dir.glob("*.jpg"))
    if max_frames is not None and len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[min(len(frames) - 1, int(i * step + step / 2))] for i in range(max_frames)]
    return frames


def data_uri(p: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def make_sbs(left: Path, right: Path, height: int = 360) -> Path:
    """V3: side-by-side video (left=reaction, right=candidate stimulus), silent, shared
    timeline — lets a video-native model check temporal alignment directly."""
    out = DATA / "sbs" / f"{left.stem}__{right.stem}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.mp4")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(left), "-i", str(right),
         "-filter_complex",
         f"[0:v]scale=-2:{height}[l];[1:v]scale=-2:{height}[r];[l][r]hstack=inputs=2[v]",
         "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", str(tmp)],
        check=True)
    tmp.rename(out)
    return out
