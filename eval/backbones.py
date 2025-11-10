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
                subprocess.run(cmd, check=True)
                
                # Verify the frame was actually created
                if not frame_path.exists():
                    raise RuntimeError(
                        f"Frame extraction succeeded but file not created: {frame_path}. "
                        f"Video duration={duration:.3f}s, requested {num_frames} frames. "
                        f"Consider using fewer frames for short videos."
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
                completion = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.01,
                    max_tokens=64,
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
