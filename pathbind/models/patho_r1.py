"""Reference model wrapper: Patho-R1-7B (Qwen2.5-VL-7B base).

Interface
---------
    m = PathoR1(ckpt_path, gpus="0", attn_impl="sdpa")
    text = m.generate(messages, max_new=768)
    attentions, image_grid = m.forward_with_attn(messages)   # grounding only

`messages` follows the OpenAI schema built by `pathbind.prompts.build_mcq_messages`
and `build_yn_messages`, plus supports an "image" content item whose value is
either a filesystem path or a PIL.Image.

To port a new model, mimic this file: it needs `generate` (for VQA) and
`forward_with_attn` (for grounding — return per-layer attentions + patch grid).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


DEFAULT_CKPT = "/9_data/zcy/models/Patho-R1"


class PathoR1:
    def __init__(
        self,
        ckpt_path: str | Path = DEFAULT_CKPT,
        gpus: str = "0",
        attn_impl: str = "sdpa",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 16384 * 28 * 28,
    ):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpus))
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            str(ckpt_path),
            use_fast=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    # -------- VQA: text generation --------
    @torch.no_grad()
    def generate(self, messages: list[dict], max_new: int = 768) -> str:
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, use_cache=True
        )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    # -------- Grounding: forward pass with attentions --------
    @torch.no_grad()
    def forward_with_attn(self, messages: list[dict]) -> dict[str, Any]:
        """Return the state needed by `pathbind.eval_grounding.eval_one`.

        Requires `attn_impl != "flash_attention_2"` (FA2 does not expose
        per-head attentions in transformers).
        """
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=img_inputs, videos=None,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model(
            **inputs, output_attentions=True, return_dict=True, use_cache=False
        )
        tok = self.processor.tokenizer
        vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
        ids = inputs["input_ids"][0].tolist()
        try:
            img_start = ids.index(vision_start_id) + 1
            img_end = ids.index(vision_end_id, img_start)
        except ValueError:
            raise RuntimeError("Vision start/end tokens not found in prompt.")
        # Qwen2.5-VL: image_grid_thw = [t, h, w] where h*w = image-token count
        # (after 2x2 spatial merge → Hp = h//2, Wp = w//2 in patch coords).
        thw = inputs["image_grid_thw"][0].tolist()
        Hp = thw[1] // 2
        Wp = thw[2] // 2
        return {
            "input_ids": inputs["input_ids"],
            "attentions": outputs.attentions,   # tuple[num_layers] of (1, H, T, T)
            "img_span": (img_start, img_end),
            "patch_grid": (Hp, Wp),
        }


# Convenience: default entry point
def load_model(ckpt_path: str = DEFAULT_CKPT, gpus: str = "0",
               attn_impl: str = "sdpa") -> PathoR1:
    return PathoR1(ckpt_path=ckpt_path, gpus=gpus, attn_impl=attn_impl)
