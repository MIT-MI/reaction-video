"""Model backbones. Common currency: OpenAI-style content parts
[{"type":"text","text":...} | {"type":"image_url","image_url":{"url": data_uri}}].

complete(parts) -> (text, in_tokens, out_tokens). Every backbone is deterministic-ish
(temperature 0 where supported) and returns real token usage for the cost ledger.

- OpenAIBackbone: gated by OPENAI_SPEND_OK=1 (owner confirms spend with co-author first).
- GeminiVertexBackbone: google-genai against Vertex AI (GOOGLE_CLOUD_PROJECT/LOCATION env).
- TinkerBackbone: any Tinker VLM/LLM ("tinker:Qwen/Qwen3.6-35B-A3B"); grant-billed.
"""

from __future__ import annotations

import base64
import os
from functools import cached_property
from pathlib import Path


class SpendNotConfirmed(RuntimeError):
    pass


def get_backbone(model: str):
    if model.startswith("hf:"):
        return HFLocalBackbone(model[3:])
    if model.startswith("tinker:"):
        return TinkerBackbone(model.split(":", 1)[1])
    if model.startswith("gemini"):
        return GeminiVertexBackbone(model)
    return OpenAIBackbone(model)


class OpenAIBackbone:
    def __init__(self, model: str = "gpt-4o"):
        if os.environ.get("OPENAI_SPEND_OK") != "1":
            raise SpendNotConfirmed(
                "OpenAI spend not confirmed. Set OPENAI_SPEND_OK=1 after the co-author sign-off.")
        self.model = model

    @cached_property
    def _client(self):
        from openai import OpenAI
        return OpenAI()

    def complete(self, parts: list[dict], max_tokens: int = 128) -> tuple[str, int, int]:
        content = []
        for p in parts:
            if p["type"] == "image_url":
                content.append({"type": "image_url",
                                "image_url": {"url": p["image_url"]["url"], "detail": "low"}})
            else:
                content.append(p)
        kwargs: dict = {"model": self.model, "messages": [{"role": "user", "content": content}]}
        if self.model.startswith(("gpt-5", "o")):
            # Reasoning models: max_completion_tokens includes reasoning tokens (billed as
            # output), fixed temperature. Minimal effort — benchmark answers, not essays.
            kwargs.update(max_completion_tokens=max(max_tokens + 512, 1024),
                          reasoning_effort="minimal")
        else:
            kwargs.update(max_tokens=max_tokens, temperature=0)
        r = self._client.chat.completions.create(**kwargs)
        u = r.usage
        return r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens


class GeminiVertexBackbone:
    def __init__(self, model: str = "gemini-3.7-flash"):
        self.model = model

    @cached_property
    def _genai(self):
        from google import genai
        return genai

    @cached_property
    def _client(self):
        # Reads GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION / GOOGLE_GENAI_USE_VERTEXAI.
        return self._genai.Client(vertexai=True)

    def complete(self, parts: list[dict], max_tokens: int = 128) -> tuple[str, int, int]:
        t = self._genai.types
        contents = []
        for p in parts:
            if p["type"] == "image_url":
                b64 = p["image_url"]["url"].split(",", 1)[1]
                contents.append(t.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg"))
            elif p["type"] == "video":
                # E1 native-video arm: raw mp4 bytes (only Gemini supports this input)
                contents.append(t.Part.from_bytes(data=Path(p["path"]).read_bytes(),
                                                  mime_type="video/mp4"))
            else:
                contents.append(p["text"])
        # max_output_tokens includes thinking tokens on Gemini 3.x; thinking_level="low"
        # suppresses them (verified: thoughts=None) so the budget goes to the visible answer.
        # Thinking cannot be fully disabled: even at "low", Flash spends ~5-100 thought
        # tokens (more on video inputs) and Pro ~70-300, all inside max_output_tokens.
        # Give headroom so the visible answer survives; thoughts bill as output but are tiny.
        budget = max_tokens + (2048 if "pro" in self.model else 512)
        try:
            cfg = t.GenerateContentConfig(
                max_output_tokens=budget, temperature=0,
                thinking_config=t.ThinkingConfig(thinking_level="low"))
            r = self._client.models.generate_content(model=self.model, contents=contents, config=cfg)
        except Exception as e:
            if "thinking" not in str(e).lower():
                raise
            cfg = t.GenerateContentConfig(max_output_tokens=max_tokens + 4096, temperature=0)
            r = self._client.models.generate_content(model=self.model, contents=contents, config=cfg)
        u = r.usage_metadata
        return r.text or "", u.prompt_token_count or 0, u.candidates_token_count or 0


class TinkerBackbone:
    def __init__(self, model: str, renderer_name: str | None = None):
        self.model = model
        self.renderer_name = renderer_name
        if not os.environ.get("TINKER_API_KEY") and os.environ.get("TINKER_API_KEY1"):
            os.environ["TINKER_API_KEY"] = os.environ["TINKER_API_KEY1"]

    @cached_property
    def _stack(self):
        import tinker
        from tinker_cookbook import model_info
        from tinker_cookbook.image_processing_utils import get_image_processor
        from tinker_cookbook.renderers import get_renderer, get_text_content
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        name = self.renderer_name or model_info.get_recommended_renderer_name(self.model)
        # Benchmark answers must be direct: disable thinking where the family supports it.
        tok = get_tokenizer(self.model)
        try:
            ip = get_image_processor(self.model)
        except Exception:
            ip = None  # text-only model (e.g. DeepSeek judge) — no image processor
        try:
            renderer = get_renderer(f"{name}_disable_thinking", tokenizer=tok, image_processor=ip)
        except Exception:
            renderer = get_renderer(name, tokenizer=tok, image_processor=ip)
        sampling = tinker.ServiceClient().create_sampling_client(base_model=self.model)
        return {"tinker": tinker, "renderer": renderer, "sampling": sampling,
                "get_text": get_text_content}

    @staticmethod
    def _to_content(parts: list[dict]) -> list[dict]:
        import base64 as b64
        import io
        from PIL import Image
        content = []
        for p in parts:
            if p["type"] == "image_url":
                raw = b64.b64decode(p["image_url"]["url"].split(",", 1)[1])
                content.append({"type": "image", "image": Image.open(io.BytesIO(raw)).convert("RGB")})
            else:
                content.append({"type": "text", "text": p["text"]})
        return content

    def complete(self, parts: list[dict], max_tokens: int = 128) -> tuple[str, int, int]:
        s = self._stack
        messages = [{"role": "user", "content": self._to_content(parts)}]
        prompt = s["renderer"].build_generation_prompt(messages)
        resp = s["sampling"].sample(
            prompt=prompt, num_samples=1,
            sampling_params=s["tinker"].SamplingParams(
                max_tokens=max_tokens, temperature=0.0,
                stop=s["renderer"].get_stop_sequences())).result()
        seq = resp.sequences[0]
        message, _ = s["renderer"].parse_response(seq.tokens)
        return s["get_text"](message), prompt.length, len(seq.tokens)


class HFLocalBackbone:
    """Local transformers inference for the fine-tune comparison (PLAN §5).
    Model spec: "hf:<base_or_path>" or "hf:<base>+<lora_adapter_dir>". Greedy decoding, same
    frame parts as every other backbone. Base AND LoRA must both be evaluated through this
    class so the with/without contrast has no Tinker-vs-local confound."""

    def __init__(self, spec: str):
        self.base, _, self.adapter = spec.partition("+")
        self.model_name = spec

    @cached_property
    def _stack(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        proc = AutoProcessor.from_pretrained(self.base)
        model = AutoModelForImageTextToText.from_pretrained(
            self.base, torch_dtype=torch.bfloat16, device_map="auto")
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter)
        model.eval()
        return {"proc": proc, "model": model, "torch": torch}

    def complete(self, parts: list[dict], max_tokens: int = 128) -> tuple[str, int, int]:
        import base64 as b64
        import io
        from PIL import Image
        s = self._stack
        images, content = [], []
        for p in parts:
            if p["type"] == "image_url":
                raw = b64.b64decode(p["image_url"]["url"].split(",", 1)[1])
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                content.append({"type": "image"})
            else:
                content.append({"type": "text", "text": p["text"]})
        text = s["proc"].apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        enc = s["proc"](text=[text], images=images or None, return_tensors="pt").to(s["model"].device)
        with s["torch"].no_grad():
            out = s["model"].generate(**enc, max_new_tokens=max_tokens, do_sample=False)
        n_in = enc["input_ids"].shape[1]
        gen = out[0][n_in:]
        return s["proc"].decode(gen, skip_special_tokens=True), int(n_in), int(gen.shape[0])
