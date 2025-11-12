from __future__ import annotations

import abc
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ScoreResult = Tuple[Optional[float], str]


def calculate_text_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Calculate the number of tokens in a text string using tiktoken."""
    try:
        import tiktoken
    except ImportError:
        # Fallback to rough estimate: ~4 chars per token
        logging.warning("tiktoken not installed. Using rough estimate for token count.")
        return len(text) // 4
    
    try:
        # Get encoding for the model
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to cl100k_base (used by gpt-4, gpt-3.5-turbo, etc.)
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def calculate_image_tokens(image_path: Path, detail: str = "high") -> int:
    """
    Calculate image tokens based on OpenAI's vision API methodology.
    
    For detail="low": 85 tokens per image
    For detail="high": 85 base + tokens for tiles based on image dimensions
    """
    if detail == "low":
        return 85
    
    try:
        from PIL import Image
    except ImportError:
        logging.warning("PIL not installed. Using default estimate of 765 tokens per image.")
        return 765
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Scale image to fit within 2048x2048 square while maintaining aspect ratio
        max_dim = 2048
        if width > max_dim or height > max_dim:
            scale = max_dim / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
        
        # Scale such that the shortest side is 768px
        min_dim = 768
        scale = min_dim / min(width, height)
        width = int(width * scale)
        height = int(height * scale)
        
        # Calculate number of 512px tiles
        tiles_width = (width + 511) // 512
        tiles_height = (height + 511) // 512
        num_tiles = tiles_width * tiles_height
        
        # Each tile is 170 tokens, plus 85 base tokens
        return 85 + (170 * num_tiles)
    except Exception as e:
        logging.warning(f"Error calculating image tokens: {e}. Using default estimate.")
        return 765


def calculate_video_tokens(video_path: Path, num_frames: int = None, detail: str = "high") -> int:
    """
    Calculate video tokens by estimating frame count and treating each as an image.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample (if None, estimates from video)
        detail: Image detail level ("low" or "high")
    """
    if num_frames is None:
        # Try to estimate frames from video
        try:
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,duration",
                "-of", "csv=p=0",
                str(video_path)
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            parts = result.stdout.strip().split(',')
            
            if len(parts) >= 2 and parts[0] and parts[1]:
                # Parse frame rate
                frame_rate_str = parts[0]
                if '/' in frame_rate_str:
                    num, denom = map(float, frame_rate_str.split('/'))
                    fps = num / denom if denom != 0 else 30
                else:
                    fps = float(frame_rate_str)
                
                duration = float(parts[1])
                # Estimate reasonable frame sampling (max 1 frame per second for long videos)
                num_frames = min(int(duration), int(duration * min(fps, 1)))
                num_frames = max(1, min(num_frames, 100))  # Cap at 100 frames
            else:
                num_frames = 8  # Default fallback
        except Exception as e:
            logging.debug(f"Error estimating video frames: {e}. Using default of 8 frames.")
            num_frames = 8
    
    # Each frame is treated as an image
    tokens_per_frame = 85 if detail == "low" else 170  # Rough average for high detail
    return num_frames * tokens_per_frame


@dataclass
class BackboneConfig:
    """Common configuration shared across backbones."""

    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    request_timeout: int = 120
    mode: str = "video"  # "video" or "image"
    num_frames: int = 8  # Number of frames to sample in image mode


class VideoQueryBackbone(abc.ABC):
    """Base class for querying video-language models for intensity scores."""

    score_regex = re.compile(r"-?\d+(?:\.\d+)?")

    def __init__(self, config: BackboneConfig):
        self.config = config

    def score_clip(self, video_path: Path, prompt: str) -> ScoreResult:
        """Runs inference and returns (score, raw_response_text)."""
        response_text = self._run_model(video_path, prompt)
        if not isinstance(response_text, str):
            response_text = str(response_text)
        score = self._parse_score(response_text)
        return score, response_text

    def _calculate_request_tokens(self, video_path: Path, prompt: str, num_frames: int = None) -> int:
        """
        Calculate estimated tokens for the request before sending.
        
        Args:
            video_path: Path to the video file
            prompt: Text prompt
            num_frames: Number of frames (for image mode or video estimation)
        
        Returns:
            Estimated token count
        """
        text_tokens = calculate_text_tokens(prompt, self.config.model_name)
        
        if self.config.mode == "video":
            # For video mode, estimate based on video properties
            video_tokens = calculate_video_tokens(
                video_path, 
                num_frames=num_frames or self.config.num_frames
            )
            total_tokens = text_tokens + video_tokens
        elif self.config.mode == "image":
            # For image mode, calculate tokens for each frame
            frame_count = num_frames or self.config.num_frames
            # Rough estimate: 170 tokens per frame on average
            video_tokens = frame_count * 170
            total_tokens = text_tokens + video_tokens
        else:
            total_tokens = text_tokens
        
        return total_tokens

    def _parse_score(self, text: str) -> Optional[float]:
        """Extracts the first numeric token from the response."""
        if not text:
            return None

        match = self.score_regex.search(text)
        if not match:
            logging.debug("Failed to parse score from response: %s", text)
            return None

        try:
            return float(match.group())
        except ValueError:
            logging.debug("Score matched but failed to convert to float: %s", match.group())
            return None

    def _extract_frames(self, video_path: Path, num_frames: int) -> List[Path]:
        """Extracts evenly-spaced frames from a video using ffmpeg."""
        temp_dir = Path(tempfile.mkdtemp(prefix="frame_extraction_"))
        frame_paths: List[Path] = []
        
        try:
            # Get video duration and frame count using ffprobe
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames,r_frame_rate,duration",
                "-of", "csv=p=0",
                str(video_path)
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            parts = result.stdout.strip().split(',')
            
            # Parse outputs: r_frame_rate, duration, nb_read_frames
            fps = None
            duration = None
            actual_frame_count = None
            
            try:
                if len(parts) >= 1 and parts[0]:
                    # Parse frame rate (e.g., "30/1" or "30000/1001")
                    frame_rate_str = parts[0]
                    if '/' in frame_rate_str:
                        num, denom = map(float, frame_rate_str.split('/'))
                        fps = num / denom if denom != 0 else None
                    else:
                        fps = float(frame_rate_str)
                
                if len(parts) >= 2 and parts[1]:
                    duration = float(parts[1])
                
                if len(parts) >= 3 and parts[2]:
                    actual_frame_count = int(parts[2])
            except (ValueError, ZeroDivisionError) as e:
                logging.debug(f"Error parsing ffprobe output: {e}")
            
            # Fallback to simple duration query if needed
            if duration is None:
                probe_cmd_simple = [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path)
                ]
                result = subprocess.run(probe_cmd_simple, capture_output=True, text=True, check=True)
                duration = float(result.stdout.strip())
            
            # If we couldn't get frame count but have fps and duration, estimate it
            if actual_frame_count is None and fps and duration:
                actual_frame_count = int(fps * duration)
            
            # Conservative fallback: estimate based on duration with minimum fps
            # Most videos are at least 24fps, so use that as a lower bound
            if actual_frame_count is None and duration:
                estimated_frame_count = int(duration * 24)  # Conservative estimate
                logging.warning(
                    f"Could not determine exact frame count for {video_path}. "
                    f"Using conservative estimate of {estimated_frame_count} frames based on duration."
                )
                actual_frame_count = estimated_frame_count
            
            # Limit num_frames to what's available, with a safety margin
            if actual_frame_count is not None:
                # Use 90% of available frames as safety margin
                safe_frame_count = int(actual_frame_count * 0.9)
                if safe_frame_count < num_frames:
                    logging.warning(
                        f"Video {video_path} has ~{actual_frame_count} frames, "
                        f"but {num_frames} were requested. Limiting to {safe_frame_count} for safety."
                    )
                    num_frames = max(1, safe_frame_count)
            
            # Additional safety check for very short videos
            # Ensure minimum time spacing between frames to avoid extraction issues
            if duration:
                # Require at least 0.3 seconds per frame for reliability
                # This ensures we don't try to extract frames too close together
                min_seconds_per_frame = 0.3
                max_frames_based_on_duration = max(1, int(duration / min_seconds_per_frame))
                
                if num_frames > max_frames_based_on_duration:
                    logging.warning(
                        f"Video {video_path} duration is {duration:.3f}s. "
                        f"Reducing frames from {num_frames} to {max_frames_based_on_duration} "
                        f"to ensure {min_seconds_per_frame}s minimum spacing between frames."
                    )
                    num_frames = max_frames_based_on_duration
            
            # Calculate frame timestamps evenly distributed
            if num_frames == 1:
                timestamps = [duration / 2.0]
            else:
                # Use 95% of duration to avoid seeking too close to the end
                effective_duration = duration * 0.95
                timestamps = [effective_duration * i / (num_frames - 1) for i in range(num_frames)]
            
            # Extract frames at specific timestamps
            for idx, timestamp in enumerate(timestamps):
                frame_path = temp_dir / f"frame_{idx:04d}.jpg"
                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-ss", f"{timestamp:.3f}",
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(frame_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Verify the frame was actually created
                if not frame_path.exists():
                    stderr_msg = result.stderr.strip() if result.stderr else "No error output"
                    raise RuntimeError(
                        f"Frame extraction failed to create file: {frame_path}. "
                        f"Video: {video_path}, duration={duration:.3f}s, "
                        f"timestamp={timestamp:.3f}s, requested {num_frames} frames. "
                        f"FFmpeg stderr: {stderr_msg}"
                    )
                frame_paths.append(frame_path)
            
            return frame_paths
        except Exception as e:
            # Clean up on error
            for frame_path in frame_paths:
                frame_path.unlink(missing_ok=True)
            try:
                temp_dir.rmdir()
            except OSError:
                pass  # Directory might not be empty or already deleted
            raise RuntimeError(f"Failed to extract frames from {video_path}: {e}") from e

    @abc.abstractmethod
    def _run_model(self, video_path: Path, prompt: str) -> str:
        """Executes the API request and returns the raw text response."""


class VLLMBackbone(VideoQueryBackbone):
    """Backbone that talks to a vLLM server via the OpenAI-compatible API."""

    def __init__(self, config: BackboneConfig):
        if not config.base_url:
            raise ValueError("vLLM backbone requires base_url pointing at the server.")
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("Please install the 'openai' package to use the vLLM backbone.") from err

        self.client = OpenAI(api_key=config.api_key or "EMPTY", base_url=config.base_url)

    def _is_internvideo_model(self) -> bool:
        """Check if the model is an InternVideo variant."""
        model_name_lower = self.config.model_name.lower()
        return "internvideo" in model_name_lower

    def _run_model(self, video_path: Path, prompt: str) -> str:
        if self.config.mode == "video":
            # Calculate and log estimated tokens
            estimated_tokens = self._calculate_request_tokens(video_path, prompt)
            logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens}")
            
            video_uri = video_path.resolve().as_uri()
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": video_uri}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0.01,
                max_tokens=32,
                timeout=self.config.request_timeout,
            )
            message = completion.choices[0].message.content
            if isinstance(message, list):
                # The OpenAI SDK sometimes returns a list of content parts.
                return " ".join(part.get("text", "") for part in message if isinstance(part, dict))
            return message or ""
        elif self.config.mode == "image":
            frame_paths = self._extract_frames(video_path, self.config.num_frames)
            try:
                # Calculate and log estimated tokens
                estimated_tokens = self._calculate_request_tokens(video_path, prompt, len(frame_paths))
                logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens} ({len(frame_paths)} frames)")
                
                # Build content with image URLs
                content = []
                for frame_path in frame_paths:
                    frame_uri = frame_path.resolve().as_uri()
                    content.append({"type": "image_url", "image_url": {"url": frame_uri}})
                
                # InternVideo models require special frame prefix format
                if self._is_internvideo_model():
                    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(frame_paths))])
                    prompt_with_prefix = video_prefix + prompt
                else:
                    prompt_with_prefix = prompt
                
                content.append({"type": "text", "text": prompt_with_prefix})
                
                completion = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.01,
                    max_tokens=32,
                    timeout=self.config.request_timeout,
                )
                message = completion.choices[0].message.content
                if isinstance(message, list):
                    return " ".join(part.get("text", "") for part in message if isinstance(part, dict))
                return message or ""
            finally:
                # Clean up extracted frames
                for frame_path in frame_paths:
                    frame_path.unlink(missing_ok=True)
                if frame_paths:
                    frame_paths[0].parent.rmdir()
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}. Use 'video' or 'image'.")


class OpenAIBackbone(VideoQueryBackbone):
    """Backbone that calls OpenAI's hosted models."""

    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("OpenAI backbone requires an API key.")
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("Please install the 'openai' package to use the OpenAI backbone.") from err

        kwargs: Dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)

    def _run_model(self, video_path: Path, prompt: str) -> str:
        if self.config.mode == "video":
            # Calculate and log estimated tokens
            estimated_tokens = self._calculate_request_tokens(video_path, prompt)
            logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens}")
            
            upload = self._upload_video(video_path)
            try:
                response = self.client.responses.create(
                    model=self.config.model_name,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_file",
                                    "file_id": upload.id,
                                },
                            ],
                        }
                    ],
                    temperature=0.01,
                    max_output_tokens=64,
                    timeout=self.config.request_timeout,
                )
            finally:
                self._delete_upload(upload.id)
            return self._extract_response_text(response)
        elif self.config.mode == "image":
            frame_paths = self._extract_frames(video_path, self.config.num_frames)
            try:
                # Calculate and log estimated tokens
                estimated_tokens = self._calculate_request_tokens(video_path, prompt, len(frame_paths))
                logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens} ({len(frame_paths)} frames)")
                
                # Build content with base64-encoded images
                import base64
                
                content = []
                for frame_path in frame_paths:
                    with frame_path.open("rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    })
                content.append({"type": "text", "text": prompt})
                
                # Use Chat Completions API for images (not Responses API)
                logging.info(f"OpenAI request: model={self.config.model_name}, frames={len(frame_paths)}, mode=image")
                completion = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.01,
                    max_tokens=64,
                    timeout=self.config.request_timeout,
                )
                
                # Log actual token usage from API response
                if hasattr(completion, 'usage') and completion.usage:
                    logging.info(
                        f"OpenAI API usage (image mode): prompt_tokens={completion.usage.prompt_tokens}, "
                        f"completion_tokens={completion.usage.completion_tokens}, "
                        f"total_tokens={completion.usage.total_tokens}"
                    )
                
                # Handle None or empty choices
                if not completion.choices or len(completion.choices) == 0:
                    logging.error(f"No choices returned from API for {video_path.name}")
                    return ""
                
                message = completion.choices[0].message.content
                if message is None:
                    logging.error(f"Message content is None for {video_path.name}")
                    return ""
                if isinstance(message, list):
                    return " ".join(part.get("text", "") for part in message if isinstance(part, dict))
                return message or ""
            finally:
                # Clean up extracted frames
                for frame_path in frame_paths:
                    frame_path.unlink(missing_ok=True)
                if frame_paths:
                    frame_paths[0].parent.rmdir()
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}. Use 'video' or 'image'.")

    def _upload_video(self, video_path: Path):
        with video_path.open("rb") as video_file:
            return self.client.files.create(
                file=video_file,
                purpose="vision",
                timeout=self.config.request_timeout,
            )

    def _delete_upload(self, file_id: str) -> None:
        try:
            self.client.files.delete(file_id)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logging.debug("Failed to delete OpenAI upload %s: %s", file_id, exc)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if not response:
            return ""

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text

        output_items = getattr(response, "output", None)
        if output_items:
            fragments: List[str] = []
            for item in output_items:
                item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
                if item_type != "message":
                    continue
                contents = getattr(item, "content", None)
                if contents is None and isinstance(item, dict):
                    contents = item.get("content")
                if not contents:
                    continue
                for content in contents:
                    content_type = getattr(content, "type", None) or (content.get("type") if isinstance(content, dict) else None)
                    if content_type == "output_text":
                        text_value = getattr(content, "text", None)
                        if text_value is None and isinstance(content, dict):
                            text_value = content.get("text")
                        if text_value:
                            fragments.append(text_value)
            if fragments:
                return "".join(fragments)

        return str(response)


class InternVideoBackbone(VideoQueryBackbone):
    """Backbone that uses InternVideo 2.5 models via transformers."""

    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        try:
            import torch
            import torchvision.transforms as T
            from decord import VideoReader, cpu
            from PIL import Image
            from torchvision.transforms.functional import InterpolationMode
            from transformers import AutoModel, AutoTokenizer
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "Install 'torch', 'torchvision', 'decord', 'transformers', and 'numpy' "
                "to use the InternVideo backbone."
            ) from err

        self.torch = torch
        self.T = T
        self.VideoReader = VideoReader
        self.cpu = cpu
        self.Image = Image
        self.InterpolationMode = InterpolationMode
        self.np = np

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            config.model_name, trust_remote_code=True
        ).half().cuda().to(torch.bfloat16)

        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

    def _build_transform(self, input_size: int):
        """Builds image transformation pipeline."""
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = self.T.Compose([
            self.T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            self.T.Resize((input_size, input_size), interpolation=self.InterpolationMode.BICUBIC),
            self.T.ToTensor(),
            self.T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Finds the closest aspect ratio from target ratios."""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        """Dynamically preprocesses image with aspect ratio preservation."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # Calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # Split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _load_video(self, video_path: Path, num_segments: int, input_size=448, max_num=1):
        """Loads and preprocesses video frames."""
        vr = self.VideoReader(str(video_path), ctx=self.cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = self._build_transform(input_size=input_size)

        # Calculate frame indices evenly distributed
        start_idx = 0
        end_idx = max_frame
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = self.np.array([
            int(start_idx + (seg_size / 2) + self.np.round(seg_size * idx))
            for idx in range(num_segments)
        ])

        for frame_index in frame_indices:
            img = self.Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = self._dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = self.torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = self.torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def _run_model(self, video_path: Path, prompt: str) -> str:
        """Runs InternVideo 2.5 model inference."""
        # Use num_frames from config as num_segments
        num_segments = self.config.num_frames
        # Clamp to reasonable range (128-512 as per official code)
        num_segments = max(128, min(512, num_segments))
        
        # Calculate and log estimated tokens
        estimated_tokens = self._calculate_request_tokens(video_path, prompt, num_segments)
        logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens} ({num_segments} segments)")

        with self.torch.no_grad():
            pixel_values, num_patches_list = self._load_video(
                video_path, num_segments=num_segments, max_num=1
            )
            pixel_values = pixel_values.to(self.torch.bfloat16).to(self.model.device)
            
            # Build video prefix as per official code
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
            question = video_prefix + prompt

            generation_config = dict(
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,  # Match the short response requirement
                top_p=0.1,
                num_beams=1
            )

            output, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True
            )
            return output


class MiniCPMBackbone(VideoQueryBackbone):
    """Backbone for MiniCPM-V-4.5 with temporal_ids support for 3D-resampler."""

    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        
        # Load model and tokenizer with vLLM mode support
        if config.base_url:
            # Use vLLM server mode - only needs OpenAI client
            try:
                from openai import OpenAI
            except ImportError as err:
                raise ImportError("Install 'openai' package for vLLM mode.") from err
            self.client = OpenAI(api_key=config.api_key or "EMPTY", base_url=config.base_url)
            self.vllm_mode = True
            logging.info("MiniCPM backbone using vLLM server at %s", config.base_url)
        else:
            # Load model locally - requires full dependencies
            try:
                import torch
                from PIL import Image
                from transformers import AutoModel, AutoTokenizer
                from decord import VideoReader, cpu
                from scipy.spatial import cKDTree
                import numpy as np
                import math
            except ImportError as err:
                raise ImportError(
                    "Install 'torch', 'transformers', 'decord', 'scipy', 'numpy' "
                    "to use the MiniCPM backbone in local mode."
                ) from err

            self.torch = torch
            self.Image = Image
            self.VideoReader = VideoReader
            self.cpu = cpu
            self.cKDTree = cKDTree
            self.np = np
            self.math = math

            # Load model locally
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                attn_implementation='sdpa',
                torch_dtype=torch.bfloat16
            ).eval().cuda()
            self.vllm_mode = False
            logging.info("MiniCPM backbone loaded model locally: %s", config.model_name)

        # MiniCPM-V-4.5 video processing parameters
        self.MAX_NUM_FRAMES = 180
        self.MAX_NUM_PACKING = 3
        self.TIME_SCALE = 0.1
        # Use config.num_frames as FPS for sampling (default 8 -> reasonable for most videos)
        self.choose_fps = max(3, min(10, config.num_frames))

    def _map_to_nearest_scale(self, values, scale):
        """Maps timestamps to nearest scale values for temporal IDs."""
        tree = self.cKDTree(self.np.asarray(scale)[:, None])
        _, indices = tree.query(self.np.asarray(values)[:, None])
        return self.np.asarray(scale)[indices]

    def _group_array(self, arr, size):
        """Groups array into chunks of given size."""
        return [arr[i:i+size] for i in range(0, len(arr), size)]

    def _encode_video(self, video_path: Path, force_packing=None):
        """
        Encodes video with temporal IDs for 3D-resampler.
        Returns (frames, temporal_id_groups).
        """
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = self.VideoReader(str(video_path), ctx=self.cpu(0))
        fps = vr.get_avg_fps()
        video_duration = len(vr) / fps

        # Calculate packing and frame count
        if self.choose_fps * int(video_duration) <= self.MAX_NUM_FRAMES:
            packing_nums = 1
            choose_frames = round(
                min(self.choose_fps, round(fps)) * min(self.MAX_NUM_FRAMES, video_duration)
            )
        else:
            packing_nums = self.math.ceil(
                video_duration * self.choose_fps / self.MAX_NUM_FRAMES
            )
            if packing_nums <= self.MAX_NUM_PACKING:
                choose_frames = round(video_duration * self.choose_fps)
            else:
                choose_frames = round(self.MAX_NUM_FRAMES * self.MAX_NUM_PACKING)
                packing_nums = self.MAX_NUM_PACKING

        if force_packing:
            packing_nums = min(force_packing, self.MAX_NUM_PACKING)

        frame_idx = [i for i in range(0, len(vr))]
        frame_idx = self.np.array(uniform_sample(frame_idx, choose_frames))

        logging.debug(
            f"Video {video_path.name}: duration={video_duration:.2f}s, "
            f"frames={len(frame_idx)}, packing={packing_nums}"
        )

        # Extract frames
        frames = vr.get_batch(frame_idx).asnumpy()

        # Calculate temporal IDs
        frame_idx_ts = frame_idx / fps
        scale = self.np.arange(0, video_duration, self.TIME_SCALE)
        frame_ts_id = self._map_to_nearest_scale(frame_idx_ts, scale) / self.TIME_SCALE
        frame_ts_id = frame_ts_id.astype(self.np.int32)

        assert len(frames) == len(frame_ts_id)

        # Convert to PIL Images
        frames = [
            self.Image.fromarray(v.astype('uint8')).convert('RGB')
            for v in frames
        ]
        frame_ts_id_group = self._group_array(frame_ts_id, packing_nums)

        return frames, frame_ts_id_group

    def _run_model(self, video_path: Path, prompt: str) -> str:
        """Runs MiniCPM-V-4.5 inference with temporal_ids support."""
        if self.vllm_mode:
            # vLLM mode: Use OpenAI API (note: temporal_ids not supported)
            logging.warning(
                "vLLM mode does not support temporal_ids parameter. "
                "For optimal MiniCPM-V-4.5 performance, use local mode (no --base-url)."
            )
            
            # Calculate and log estimated tokens
            estimated_tokens = self._calculate_request_tokens(video_path, prompt)
            logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens}")
            
            video_uri = video_path.resolve().as_uri()
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": video_uri}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0.01,
                max_tokens=32,
                timeout=self.config.request_timeout,
            )
            message = completion.choices[0].message.content
            if isinstance(message, list):
                return " ".join(
                    part.get("text", "") for part in message if isinstance(part, dict)
                )
            return message or ""
        else:
            # Local mode: Use transformers with temporal_ids
            frames, frame_ts_id_group = self._encode_video(video_path)
            
            # Calculate and log estimated tokens
            estimated_tokens = self._calculate_request_tokens(video_path, prompt, len(frames))
            logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens} ({len(frames)} frames)")

            msgs = [
                {'role': 'user', 'content': frames + [prompt]},
            ]

            with self.torch.no_grad():
                answer = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    use_image_id=False,
                    max_slice_nums=1,
                    temporal_ids=frame_ts_id_group,
                    max_new_tokens=32,
                    sampling=False,
                )
            return answer


class VideoChatFlashBackbone(VideoQueryBackbone):
    """Backbone for VideoChat-Flash models with hierarchical compression for long-context video modeling."""

    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        
        # Load model and tokenizer with vLLM mode support
        if config.base_url:
            # Use vLLM server mode - only needs OpenAI client
            try:
                from openai import OpenAI
            except ImportError as err:
                raise ImportError("Install 'openai' package for vLLM mode.") from err
            self.client = OpenAI(api_key=config.api_key or "EMPTY", base_url=config.base_url)
            self.vllm_mode = True
            logging.info("VideoChat-Flash backbone using vLLM server at %s", config.base_url)
        else:
            # Load model locally - requires full dependencies
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as err:
                raise ImportError(
                    "Install 'torch', 'transformers' to use the VideoChat-Flash backbone in local mode."
                ) from err

            self.torch = torch
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                config.model_name, trust_remote_code=True
            ).to(torch.bfloat16).cuda()
            
            self.image_processor = self.model.get_vision_tower().image_processor
            
            # Configure compression settings (optional)
            mm_llm_compress = False  # Use global compress or not
            if mm_llm_compress:
                self.model.config.mm_llm_compress = True
                self.model.config.llm_compress_type = "uniform0_attention"
                self.model.config.llm_compress_layer_list = [4, 18]
                self.model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
            else:
                self.model.config.mm_llm_compress = False
            
            self.vllm_mode = False
            logging.info("VideoChat-Flash backbone loaded model locally: %s", config.model_name)
        
        # VideoChat-Flash supports up to ~10,000 frames, but we'll use num_frames from config
        self.max_num_frames = config.num_frames if config.num_frames else 512

    def _run_model(self, video_path: Path, prompt: str) -> str:
        """Runs VideoChat-Flash inference."""
        if self.vllm_mode:
            # vLLM mode: Use OpenAI API
            # Calculate and log estimated tokens
            estimated_tokens = self._calculate_request_tokens(video_path, prompt)
            logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens}")
            
            video_uri = video_path.resolve().as_uri()
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": video_uri}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0.01,
                max_tokens=32,
                timeout=self.config.request_timeout,
            )
            message = completion.choices[0].message.content
            if isinstance(message, list):
                return " ".join(
                    part.get("text", "") for part in message if isinstance(part, dict)
                )
            return message or ""
        else:
            # Local mode: Use transformers with model.chat()
            # Calculate and log estimated tokens
            estimated_tokens = self._calculate_request_tokens(video_path, prompt, self.max_num_frames)
            logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens} (max {self.max_num_frames} frames)")
            
            generation_config = dict(
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                top_p=0.1,
                num_beams=1
            )
            
            with self.torch.no_grad():
                output, _ = self.model.chat(
                    video_path=str(video_path),
                    tokenizer=self.tokenizer,
                    user_prompt=prompt,
                    return_history=True,
                    max_num_frames=self.max_num_frames,
                    generation_config=generation_config
                )
            return output


class GeminiBackbone(VideoQueryBackbone):
    """Backbone that uses Google's Gemini API for video scoring."""

    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("Gemini backbone requires an API key.")
        try:
            import google.generativeai as genai
        except ImportError as err:
            raise ImportError("Install 'google-generativeai' to use the Gemini backbone.") from err

        genai.configure(api_key=config.api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(config.model_name)

    def _run_model(self, video_path: Path, prompt: str) -> str:
        if self.config.mode == "video":
            return self._run_model_video(video_path, prompt)
        elif self.config.mode == "image":
            return self._run_model_image(video_path, prompt)
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}. Use 'video' or 'image'.")

    def _run_model_video(self, video_path: Path, prompt: str) -> str:
        # Calculate and log estimated tokens
        estimated_tokens = self._calculate_request_tokens(video_path, prompt)
        logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens}")
        
        max_retries = 3
        for attempt in range(max_retries):
            uploaded_file = None
            try:
                uploaded_file = self._genai.upload_file(path=str(video_path))
                import time

                while uploaded_file.state.name == "PROCESSING":
                    time.sleep(2)
                    uploaded_file = self._genai.get_file(name=uploaded_file.name)

                if uploaded_file.state.name == "ACTIVE":
                    response = self._model.generate_content(
                        [uploaded_file, {"text": prompt}],
                        request_options={"timeout": self.config.request_timeout},
                    )

                    if hasattr(response, "text") and response.text is not None:
                        return response.text
                    if hasattr(response, "candidates"):
                        for candidate in response.candidates or []:
                            parts = getattr(candidate, "content", None)
                            if not parts:
                                continue
                            text_fragments = []
                            for part in getattr(parts, "parts", []):
                                text = getattr(part, "text", None)
                                if text:
                                    text_fragments.append(text)
                            if text_fragments:
                                return " ".join(text_fragments)
                    return str(response)

                logging.debug(
                    f"Attempt {attempt + 1} of {max_retries}: File {uploaded_file.name} is not in an ACTIVE state. It is {uploaded_file.state.name}"
                )
                if attempt == max_retries - 1:
                    raise ValueError(
                        f"File {uploaded_file.name} failed to become ACTIVE after {max_retries} attempts. Final state: {uploaded_file.state.name}"
                    )

            except Exception as e:
                logging.debug(f"An error occurred during attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
            finally:
                if uploaded_file:
                    try:
                        self._genai.delete_file(uploaded_file.name)
                    except Exception as exc:
                        logging.debug("Failed to delete Gemini upload %s: %s", uploaded_file.name, exc)
            time.sleep(5)  # Wait before retrying

        raise RuntimeError("Exhausted all retries for file upload and processing.")

    def _run_model_image(self, video_path: Path, prompt: str) -> str:
        import time
        
        frame_paths = self._extract_frames(video_path, self.config.num_frames)
        
        # Calculate and log estimated tokens
        estimated_tokens = self._calculate_request_tokens(video_path, prompt, len(frame_paths))
        logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens} ({len(frame_paths)} frames)")
        
        uploaded_files = []
        try:
            # Upload all frames
            for frame_path in frame_paths:
                uploaded_file = self._genai.upload_file(path=str(frame_path))
                uploaded_files.append(uploaded_file)
            
            # Wait for all files to be processed
            max_retries = 3
            for attempt in range(max_retries):
                all_active = True
                for idx, uploaded_file in enumerate(uploaded_files):
                    uploaded_files[idx] = self._genai.get_file(name=uploaded_file.name)
                    if uploaded_files[idx].state.name == "PROCESSING":
                        all_active = False
                
                if all_active:
                    break
                
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise RuntimeError("Some files failed to process in time")
            
            # Build content with all images and prompt
            content = []
            for uploaded_file in uploaded_files:
                if uploaded_file.state.name == "ACTIVE":
                    content.append(uploaded_file)
            content.append({"text": prompt})
            
            response = self._model.generate_content(
                content,
                request_options={"timeout": self.config.request_timeout},
            )
            
            if hasattr(response, "text") and response.text is not None:
                return response.text
            if hasattr(response, "candidates"):
                for candidate in response.candidates or []:
                    parts = getattr(candidate, "content", None)
                    if not parts:
                        continue
                    text_fragments = []
                    for part in getattr(parts, "parts", []):
                        text = getattr(part, "text", None)
                        if text:
                            text_fragments.append(text)
                    if text_fragments:
                        return " ".join(text_fragments)
            return str(response)
        finally:
            # Clean up uploaded files and local frames
            for uploaded_file in uploaded_files:
                try:
                    self._genai.delete_file(uploaded_file.name)
                except Exception as exc:
                    logging.debug("Failed to delete Gemini upload %s: %s", uploaded_file.name, exc)
            for frame_path in frame_paths:
                frame_path.unlink(missing_ok=True)
            if frame_paths:
                frame_paths[0].parent.rmdir()


class OpenRouterBackbone(VideoQueryBackbone):
    """
    Backbone for OpenRouter API with single-image support.
    Some OpenRouter models only support one image per request.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("OpenRouter backbone requires an API key.")
        if not config.base_url:
            config.base_url = "https://openrouter.ai/api/v1"
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("Please install the 'openai' package to use the OpenRouter backbone.") from err

        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def _run_model(self, video_path: Path, prompt: str) -> str:
        """Extract middle frame and send single image to OpenRouter."""
        # Extract only 1 frame (middle frame)
        frame_paths = self._extract_frames(video_path, num_frames=1)
        try:
            # Calculate and log estimated tokens
            estimated_tokens = self._calculate_request_tokens(video_path, prompt, 1)
            logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens} (1 frame - middle)")
            
            # Build content with single base64-encoded image
            import base64
            
            with frame_paths[0].open("rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                },
                {"type": "text", "text": prompt}
            ]
            
            # Use Chat Completions API
            logging.info(f"OpenRouter request: model={self.config.model_name}, frames=1, mode=single-image")
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0.01,
                max_tokens=64,
                timeout=self.config.request_timeout,
            )
            
            # Log actual token usage from API response
            if hasattr(completion, 'usage') and completion.usage:
                logging.info(
                    f"OpenRouter API usage: prompt_tokens={completion.usage.prompt_tokens}, "
                    f"completion_tokens={completion.usage.completion_tokens}, "
                    f"total_tokens={completion.usage.total_tokens}"
                )
            
            # Handle None or empty choices
            if not completion.choices or len(completion.choices) == 0:
                logging.error(f"No choices returned from OpenRouter API for {video_path.name}")
                return ""
            
            message = completion.choices[0].message.content
            if message is None:
                logging.error(f"Message content is None for {video_path.name}")
                return ""
            if isinstance(message, list):
                return " ".join(part.get("text", "") for part in message if isinstance(part, dict))
            return message or ""
        finally:
            # Clean up extracted frame
            for frame_path in frame_paths:
                frame_path.unlink(missing_ok=True)
            if frame_paths:
                try:
                    frame_paths[0].parent.rmdir()
                except OSError:
                    pass


class ReplicateBackbone(VideoQueryBackbone):
    """Backbone that uses Replicate API for video scoring."""

    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("Replicate backbone requires an API key.")
        try:
            import replicate
        except ImportError as err:
            raise ImportError("Install 'replicate' package to use the Replicate backbone.") from err
        
        self._replicate = replicate
        self.client = replicate.Client(api_token=config.api_key)
        
        # Get model version for predictions
        model = self.client.models.get(config.model_name)
        self.version_id = model.latest_version.id
        logging.info(f"Replicate model {config.model_name} version: {self.version_id}")

    def _run_model(self, video_path: Path, prompt: str) -> str:
        """Run model using Replicate API with video file upload."""
        if self.config.mode == "video":
            return self._run_model_video(video_path, prompt)
        elif self.config.mode == "image":
            return self._run_model_image(video_path, prompt)
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}. Use 'video' or 'image'.")

    def _run_model_video(self, video_path: Path, prompt: str) -> str:
        """Run model with video file using predictions.create()."""
        try:
            # Calculate and log estimated tokens
            estimated_tokens = self._calculate_request_tokens(video_path, prompt)
            logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens}")
            
            # Build input parameters based on model
            input_params = {
                "video": open(video_path, "rb"),
                "prompt": prompt,
            }
            
            # Add model-specific parameters
            model_lower = self.config.model_name.lower()
            if "videollama" in model_lower:
                input_params["max_new_tokens"] = 32
                input_params["temperature"] = 0.01
            elif "internvl" in model_lower:
                input_params["max_new_tokens"] = 32
                input_params["temperature"] = 0.01
            
            # Use predictions.create() API
            prediction = self.client.predictions.create(
                version=self.version_id,
                input=input_params
            )
            
            # Wait for prediction to complete
            prediction.wait()
            
            # Handle different output formats
            output = prediction.output
            if isinstance(output, str):
                return output
            elif isinstance(output, list):
                return "".join(str(item) for item in output)
            else:
                return str(output)
                
        except Exception as e:
            logging.error(f"Replicate API error: {e}")
            raise

    def _run_model_image(self, video_path: Path, prompt: str) -> str:
        """Run model with extracted frames."""
        frame_paths = self._extract_frames(video_path, self.config.num_frames)
        try:
            estimated_tokens = self._calculate_request_tokens(video_path, prompt, len(frame_paths))
            logging.info(f"Estimated input tokens for {video_path.name}: {estimated_tokens} ({len(frame_paths)} frames)")
            
            # DeepSeek-VL2 only supports single image input
            # Use the middle frame as representative
            middle_idx = len(frame_paths) // 2
            selected_frame = frame_paths[middle_idx]
            
            # Use predictions.create() API which works correctly
            with open(selected_frame, "rb") as image_file:
                prediction = self.client.predictions.create(
                    version=self.version_id,
                    input={
                        "image": image_file,
                        "prompt": prompt,
                        "max_length_tokens": 32,
                        "temperature": 0.01,
                    }
                )
            
            # Wait for prediction to complete
            prediction.wait()
            
            # Handle different output formats
            output = prediction.output
            if isinstance(output, str):
                return output
            elif isinstance(output, list):
                return "".join(str(item) for item in output)
            else:
                return str(output)
                
        except Exception as e:
            logging.error(f"Replicate API error with frames: {e}")
            raise
        finally:
            # Clean up extracted frames
            for frame_path in frame_paths:
                frame_path.unlink(missing_ok=True)
            if frame_paths:
                try:
                    frame_paths[0].parent.rmdir()
                except OSError:
                    pass


class Qwen2_5OmniBackbone(VideoQueryBackbone):
    """
    Backbone for Qwen2.5-Omni with automatic frame upsampling for short videos.
    
    Qwen2.5-Omni requires at least 32 frames. For shorter videos, this backbone
    automatically upsamples using FFmpeg frame duplication to ensure consistent input.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        if not config.base_url:
            raise ValueError("Qwen2.5-Omni backbone requires base_url pointing at the vLLM server.")
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("Please install the 'openai' package to use the Qwen2.5-Omni backbone.") from err

        self.client = OpenAI(api_key=config.api_key or "EMPTY", base_url=config.base_url)
        
        # Qwen2.5-Omni requires at least 32 frames with consistent encoding
        # Due to discrepancies between ffprobe counts and vLLM's decoder,
        # we standardize ALL videos to ensure reliable decoding
        self.MIN_FRAMES = 32
        self.TARGET_FPS = 8  # Target FPS for standardization
        self.TARGET_SIZE = 448  # Target resolution
        self.TARGET_DURATION = 5.0  # Target duration in seconds (8 fps * 5s = 40 frames, with padding = ~48 frames)
        self.ALWAYS_PREPROCESS = True  # Always standardize videos for reliability
        
        logging.info(
            "Qwen2.5-Omni backbone initialized with video standardization "
            "(always_preprocess=%s, target=%d fps, %d frames)",
            self.ALWAYS_PREPROCESS, self.TARGET_FPS, int(self.TARGET_FPS * self.TARGET_DURATION) + 8
        )

    def _get_frame_count(self, video_path: Path) -> int:
        """Get the total number of frames in a video."""
        try:
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0",
                str(video_path)
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            frame_count = int(result.stdout.strip())
            return frame_count
        except Exception as e:
            logging.warning(f"Could not determine frame count for {video_path}: {e}")
            # Fallback: estimate from duration and fps
            try:
                probe_cmd = [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate,duration",
                    "-of", "csv=p=0",
                    str(video_path)
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                parts = result.stdout.strip().split(',')
                if len(parts) >= 2 and parts[0] and parts[1]:
                    frame_rate_str = parts[0]
                    if '/' in frame_rate_str:
                        num, denom = map(float, frame_rate_str.split('/'))
                        fps = num / denom if denom != 0 else 30
                    else:
                        fps = float(frame_rate_str)
                    duration = float(parts[1])
                    return int(fps * duration)
            except Exception:
                pass
            return 0

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds."""
        try:
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logging.warning(f"Could not determine duration for {video_path}: {e}")
            return 0.0

    def _standardize_video(self, video_path: Path) -> Path:
        """
        Standardize video to consistent format for Qwen2.5-Omni.
        
        Creates a standardized video at 8 fps, 448x448, with guaranteed ≥32 frames.
        Uses adaptive padding strategy based on input duration.
        
        Returns path to the standardized video (in a temporary location).
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="qwen_omni_std_"))
        output_path = temp_dir / f"{video_path.stem}_std.mp4"
        
        try:
            # Get input video duration to determine padding strategy
            input_duration = self._get_video_duration(video_path)
            
            # Calculate how much padding we need
            # At 8 fps, we need at least 4 seconds to get 32 frames
            min_duration = self.MIN_FRAMES / self.TARGET_FPS  # 32/8 = 4 seconds
            
            if input_duration > 0 and input_duration < min_duration:
                # Short video: add more padding at the end
                padding_duration = min_duration - input_duration + 1.0  # Extra 1s for safety
                logging.debug(
                    f"Short video ({input_duration:.2f}s), adding {padding_duration:.2f}s padding"
                )
            else:
                # Normal/long video: add standard padding
                padding_duration = 1.0
            
            # FFmpeg command for standardization
            # Strategy: Resample to 8 fps, resize, then pad with cloned frames
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-vf",
                (
                    # First, scale and pad to target resolution
                    f"scale={self.TARGET_SIZE}:{self.TARGET_SIZE}:force_original_aspect_ratio=decrease,"
                    f"pad={self.TARGET_SIZE}:{self.TARGET_SIZE}:(ow-iw)/2:(oh-ih)/2:color=black,"
                    # Then add padding by cloning last frame
                    f"tpad=stop_mode=clone:stop_duration={padding_duration}"
                ),
                "-r", str(self.TARGET_FPS),  # Output at constant 8 fps
                "-vsync", "cfr",  # Constant frame rate
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                "-crf", "23",  # Quality setting
                str(output_path)
            ]
            
            logging.debug(
                f"Standardizing {video_path.name}: duration={input_duration:.2f}s, "
                f"padding={padding_duration:.2f}s, target_fps={self.TARGET_FPS}"
            )
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=120
            )
            
            # Verify output file exists and is valid
            if not output_path.exists():
                stderr_output = result.stderr if result.stderr else "No error output"
                raise RuntimeError(
                    f"FFmpeg did not create output file: {output_path}\n"
                    f"Input: {video_path}\n"
                    f"FFmpeg stderr: {stderr_output}"
                )
            
            # Check file size (should be > 0)
            file_size = output_path.stat().st_size
            if file_size == 0:
                raise RuntimeError(
                    f"Standardized video is empty (0 bytes): {output_path}\n"
                    f"Input: {video_path}"
                )
            
            # Verify the standardized video has enough frames
            std_frame_count = self._get_frame_count(output_path)
            std_duration = self._get_video_duration(output_path)
            
            if std_frame_count < self.MIN_FRAMES:
                # This is a critical error - the video won't work with Qwen2.5-Omni
                error_msg = (
                    f"Standardized video has insufficient frames!\n"
                    f"  Input: {video_path.name} (duration={input_duration:.2f}s)\n"
                    f"  Output: {output_path.name} (duration={std_duration:.2f}s, frames={std_frame_count})\n"
                    f"  Required: {self.MIN_FRAMES} frames minimum\n"
                    f"  This video will likely fail with Qwen2.5-Omni."
                )
                logging.error(error_msg)
                # Don't raise, but log the issue - maybe vLLM will be more lenient
            else:
                logging.debug(
                    f"Successfully standardized {video_path.name}: "
                    f"{std_frame_count} frames, {std_duration:.2f}s (>= {self.MIN_FRAMES} required)"
                )
            
            return output_path
            
        except subprocess.TimeoutExpired:
            error_msg = f"FFmpeg standardization timed out (>120s) for {video_path}"
            logging.error(error_msg)
            # Clean up
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            try:
                temp_dir.rmdir()
            except OSError:
                pass
            raise RuntimeError(error_msg)
            
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"FFmpeg standardization failed for {video_path}\n"
                f"Return code: {e.returncode}\n"
                f"Stderr: {e.stderr if e.stderr else 'No error output'}\n"
                f"Command: {' '.join(str(x) for x in cmd)}"
            )
            logging.error(error_msg)
            # Clean up
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            try:
                temp_dir.rmdir()
            except OSError:
                pass
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error standardizing {video_path}: {e}"
            logging.error(error_msg)
            # Clean up on error
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            try:
                temp_dir.rmdir()
            except OSError:
                pass
            raise RuntimeError(error_msg) from e

    def _run_model(self, video_path: Path, prompt: str) -> str:
        """
        Runs Qwen2.5-Omni inference with automatic video standardization.
        
        Standardizes all videos to a consistent format (8 fps, 448x448, ~48 frames)
        to ensure reliable decoding by vLLM, regardless of input format.
        """
        standardized_path = None
        temp_dir = None
        
        try:
            if self.ALWAYS_PREPROCESS:
                # Always standardize videos for reliability
                logging.info(f"Standardizing video {video_path.name} for Qwen2.5-Omni...")
                standardized_path = self._standardize_video(video_path)
                temp_dir = standardized_path.parent
                video_to_use = standardized_path
            else:
                # Legacy path: only standardize if frame count is low
                frame_count = self._get_frame_count(video_path)
                if frame_count == 0 or frame_count < self.MIN_FRAMES:
                    logging.warning(
                        f"Video {video_path.name} has {frame_count if frame_count > 0 else 'unknown'} frames. "
                        f"Standardizing..."
                    )
                    standardized_path = self._standardize_video(video_path)
                    temp_dir = standardized_path.parent
                    video_to_use = standardized_path
                else:
                    logging.info(f"Video {video_path.name} has {frame_count} frames. Using original.")
                    video_to_use = video_path
            
            # Calculate and log estimated tokens
            estimated_tokens = self._calculate_request_tokens(video_to_use, prompt)
            logging.info(f"Estimated input tokens: {estimated_tokens}")
            
            # Run inference using vLLM OpenAI-compatible API
            video_uri = video_to_use.resolve().as_uri()
            
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "video_url", "video_url": {"url": video_uri}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    temperature=0.01,
                    max_tokens=32,
                    timeout=self.config.request_timeout,
                )
                
                message = completion.choices[0].message.content
                if isinstance(message, list):
                    return " ".join(part.get("text", "") for part in message if isinstance(part, dict))
                return message or ""
                
            except Exception as api_error:
                # Enhanced error logging for debugging
                error_type = type(api_error).__name__
                
                # Try to extract more details
                error_details = {
                    "error_type": error_type,
                    "original_video": video_path.name,
                    "processed_video": video_to_use.name,
                    "video_uri": video_uri,
                }
                
                # Add frame count info if available
                try:
                    error_details["frame_count"] = self._get_frame_count(video_to_use)
                    error_details["duration"] = self._get_video_duration(video_to_use)
                    error_details["file_size_mb"] = video_to_use.stat().st_size / (1024 * 1024)
                except Exception:
                    pass
                
                logging.error(
                    f"vLLM API error for {video_path.name}:\n"
                    f"  Error: {api_error}\n"
                    f"  Details: {error_details}"
                )
                
                # Re-raise the original error
                raise
            
        finally:
            # Clean up standardized video if it was created
            if standardized_path and standardized_path.exists():
                standardized_path.unlink(missing_ok=True)
                logging.debug(f"Cleaned up standardized video: {standardized_path.name}")
            if temp_dir and temp_dir.exists():
                try:
                    temp_dir.rmdir()
                except OSError:
                    pass  # Directory might not be empty


class VideoChatR1Backbone(VideoQueryBackbone):
    """
    Backbone for VideoChat-R1_5-7B with test-time scaling via iterative perception.
    
    This model enhances spatio-temporal perception through reinforcement fine-tuning
    and supports multi-perception inference with temporal localization (glue).
    Based on Qwen2.5-VL-7B-Instruct with specialized video understanding.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        
        try:
            import torch
            import re
            import ast
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError as err:
            raise ImportError(
                "Install 'torch', 'transformers' to use the VideoChatR1 backbone."
            ) from err

        self.torch = torch
        self.re = re
        self.ast = ast
        
        # Load model and processor
        model_path = config.model_name
        logging.info(f"Loading VideoChat-R1_5 model from {model_path}...")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # Import custom vision processing utilities
        try:
            # Try to import from the installed package location
            import sys
            github_path = Path.home() / "Code" / "VideoChat-R1" / "Videochat-R1.5" / "src_eval"
            if github_path.exists():
                sys.path.insert(0, str(github_path))
            from my_vision_process import process_vision_info
            self.process_vision_info = process_vision_info
            logging.info("Successfully imported custom process_vision_info")
        except ImportError:
            # Fallback to standard qwen_vl_utils
            logging.warning(
                "Custom my_vision_process not found. Falling back to qwen_vl_utils. "
                "For optimal performance, clone https://github.com/OpenGVLab/VideoChat-R1"
            )
            try:
                from qwen_vl_utils import process_vision_info
                self.process_vision_info = process_vision_info
            except ImportError as err:
                raise ImportError(
                    "Install 'qwen-vl-utils' or clone VideoChat-R1 repository: "
                    "pip install qwen-vl-utils"
                ) from err
        
        # Number of perception iterations for test-time scaling
        # More iterations = better performance but slower inference
        self.num_perceptions = config.num_frames if config.num_frames else 3
        self.num_perceptions = max(1, min(5, self.num_perceptions))  # Clamp to 1-5
        
        # Total pixels for video processing (controls frame sampling)
        # VideoChat-R1.5 uses 128*12*28*28 for standard quality
        self.total_pixels = 128 * 12 * 28 * 28
        self.min_pixels = 128 * 28 * 28
        
        logging.info(
            f"VideoChat-R1_5 initialized with {self.num_perceptions} perception iterations"
        )

    def _build_prompt_with_glue(self, question: str, use_glue: bool) -> str:
        """Build prompt template with or without glue (temporal localization)."""
        if use_glue:
            template = """Answer the question: "{question}" according to the content of the video.

Output your think process within the <think> </think> tags.

Then, provide your answer within the <answer> </answer> tags. At the same time, in the <glue> </glue> tags, present the precise time period in seconds of the video clips on which you base your answer to this question in the format of [(s1, e1), (s2, e2), ...]. For example: <think>...</think><answer>7.5</answer><glue>[(5.2, 10.4)]</glue>."""
        else:
            template = """Answer the question: "{question}" according to the content of the video.

Output your think process within the <think> </think> tags.

Then, provide your answer within the <answer> </answer> tags. For example: <think>...</think><answer>7.5</answer>."""
        
        return template.replace("{question}", question)

    def _inference_single(
        self, 
        video_path: Path, 
        prompt: str, 
        pred_glue: Optional[List[Tuple[float, float]]] = None,
        max_new_tokens: int = 2048
    ) -> str:
        """Run single inference pass with optional temporal key_time constraint."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "key_time": pred_glue,
                        "total_pixels": self.total_pixels,
                        "min_pixels": self.min_pixels,
                    },
                    {"type": "text", "text": prompt},
                ]
            },
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs, video_kwargs = self.process_vision_info(
            messages, return_video_kwargs=True, client=None
        )
        fps_inputs = video_kwargs.get('fps')
        
        # Prepare inputs for model
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=fps_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        
        # Generate
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
        
        # Decode only the generated tokens
        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):] 
            for i in range(len(output_ids))
        ]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return output_text[0]

    def _extract_glue(self, response: str) -> Optional[List[Tuple[float, float]]]:
        """Extract temporal glue (time periods) from model response."""
        pattern_glue = r'<glue>(.*?)</glue>'
        match_glue = self.re.search(pattern_glue, response, self.re.DOTALL)
        
        if not match_glue:
            return None
        
        try:
            glue_str = match_glue.group(1).strip()
            pred_glue = self.ast.literal_eval(glue_str)
            
            # Validate format: should be list of tuples
            if isinstance(pred_glue, list):
                validated = []
                for item in pred_glue:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        s, e = float(item[0]), float(item[1])
                        if s < e:  # Valid time interval
                            validated.append((s, e))
                return validated if validated else None
            
        except Exception as e:
            logging.debug(f"Failed to parse glue from response: {e}")
        
        return None

    def _extract_answer(self, response: str) -> str:
        """Extract answer from <answer> tags, or return full response."""
        pattern_answer = r'<answer>(.*?)</answer>'
        match_answer = self.re.search(pattern_answer, response, self.re.DOTALL)
        
        if match_answer:
            return match_answer.group(1).strip()
        
        # Fallback: return everything after </think> if present
        if '</think>' in response:
            parts = response.split('</think>', 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        return response.strip()

    def _run_model(self, video_path: Path, prompt: str) -> str:
        """
        Run VideoChat-R1.5 with iterative perception for test-time scaling.
        
        The model performs multiple perception passes:
        1. Initial passes extract temporal glue (key time periods)
        2. Final pass uses accumulated glue for refined answer
        """
        # Calculate and log estimated tokens
        estimated_tokens = self._calculate_request_tokens(video_path, prompt)
        logging.info(
            f"Estimated input tokens for {video_path.name}: {estimated_tokens} "
            f"({self.num_perceptions} perception iterations)"
        )
        
        answers = []
        pred_glue = None
        
        # Iterative perception loop
        for perception_idx in range(self.num_perceptions):
            is_final = (perception_idx == self.num_perceptions - 1)
            
            # Build prompt: include glue extraction for all but final iteration
            use_glue = not is_final
            perception_prompt = self._build_prompt_with_glue(prompt, use_glue)
            
            logging.debug(
                f"Perception {perception_idx + 1}/{self.num_perceptions} "
                f"(glue={'enabled' if use_glue else 'disabled'}, "
                f"key_time={pred_glue if pred_glue else 'none'})"
            )
            
            # Run inference
            response = self._inference_single(
                video_path,
                perception_prompt,
                pred_glue=pred_glue,
                max_new_tokens=2048 if not is_final else 512
            )
            
            answers.append(response)
            
            # Extract glue for next iteration (except on final)
            if not is_final:
                new_glue = self._extract_glue(response)
                if new_glue:
                    pred_glue = new_glue
                    logging.debug(f"Extracted glue: {pred_glue}")
        
        # Return the final answer
        final_response = answers[-1]
        final_answer = self._extract_answer(final_response)
        
        logging.debug(f"Final answer: {final_answer}")
        
        return final_answer
