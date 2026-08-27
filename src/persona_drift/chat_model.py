"""Thin transformers wrapper around a single causal LM used as either the
self-chat agent or the simulated user.

Sets HF_HOME to /scratch/hcao2/hf_cache before importing transformers if the
caller hasn't already set it, so a forgotten `export HF_HOME=...` cannot
silently dump multi-GB model weights into the (small, shared) home
directory. Override by exporting HF_HOME yourself before running.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

_DEFAULT_HF_HOME = "/scratch/hcao2/hf_cache"
if "HF_HOME" not in os.environ:
    pathlib.Path(_DEFAULT_HF_HOME).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = _DEFAULT_HF_HOME

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 256
    do_sample: bool = True


class ChatModel:
    """Loads one causal LM once; each generate() call reseeds torch's RNG
    globally so trajectory-level and probe-repeat-level seeding (protocol
    section 1/2: "每条轨迹固定 seed", probe repeated with different seeds) is
    exact and reproducible, without depending on transformers-version-
    specific `generator=` kwarg support in `.generate()`."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str | None = None,
        enable_thinking: bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.enable_thinking = enable_thinking
        resolved_dtype = dtype or ("bfloat16" if device.startswith("cuda") else "float32")
        torch_dtype = getattr(torch, resolved_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            # transformers renamed torch_dtype= to dtype= around 4.56; support both.
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
        self.model.to(device)
        self.model.eval()

    def generate(self, messages: list[dict[str, str]], seed: int, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Older tokenizers / non-Qwen3 chat templates without thinking-mode support.
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.device)

        torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
