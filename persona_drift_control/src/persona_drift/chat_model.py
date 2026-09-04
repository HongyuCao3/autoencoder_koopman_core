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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


@dataclass(frozen=True)
class SteeringConfig:
    """Channel C (u_steer, DATA_COLLECTION_PROTOCOL.md section 3): adds
    alpha * direction to the residual stream at `layer` for every generated
    token, via a forward hook on that decoder layer -- see
    ChatModel._register_steering_hook. `direction` is the raw (unnormalized)
    diff-in-means vector from activation_direction.compute_safety_direction;
    its own magnitude already is the natural alpha=1 scale (the protocol's
    "alpha0"), so the sweep grid {-1, -0.5, 0, 0.5, 1} multiplies it
    directly rather than needing a separately calibrated scalar."""

    # Uses the same 1-indexed convention as hidden_state_at_layer's
    # output_hidden_states (hidden_states[0]=embeddings, hidden_states[L]=
    # output of decoder block index L-1): so a direction calibrated at
    # layer=L is steered by adding to block index L-1's output, which is
    # exactly the tensor hidden_states[L] was read from. See
    # _register_steering_hook.
    layer: int
    direction: np.ndarray  # shape (hidden_size,), points toward the harmless/safe pole
    alpha: float


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 256
    do_sample: bool = True
    # Defaults are transformers' own no-op values, so existing callers (the
    # live self-chat loop in selfchat.py, validated across the completed
    # signal-screening run) are byte-for-byte unaffected. Non-default values
    # are used by scripts/generate_user_scripts.py, whose long ungrounded
    # reference/script self-chat (no real feedback loop turn to turn) was
    # observed to collapse into near-verbatim repeated paragraphs by ~turn 10
    # without them -- see docs/experiments/signal_screening_pilot.md.
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


_THINK_END_TAG = "</think>"


def _build_prompt_text(tokenizer, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    """Renders `messages` into the exact prompt string `.generate()` and
    `next_token_logits()` both feed the tokenizer -- factored out so the two
    call sites cannot drift apart. That sameness is not cosmetic: it is the
    entire basis for `next_token_logits`'s validity as a stand-in for what
    `.generate()` would have sampled as the first token (see
    continuous_readout_plan.md's G1 gate). Falls back to the
    thinking-mode-unaware call for tokenizers whose chat template doesn't
    accept `enable_thinking=` (same TypeError fallback `generate()` used
    before this was factored out)."""

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _split_at_think_end(token_ids: list[int], think_end_id: int | None) -> tuple[list[int], list[int]]:
    """Splits generated token ids into (thinking_ids, content_ids) at the
    first </think> token (inclusive). Returns ([], token_ids) when
    think_end_id is None or doesn't occur -- covers both
    enable_thinking=False (no such token is ever emitted) and tokenizers
    without a </think> token at all. Pure/torch-free so it's unit-testable
    without a real tokenizer -- see chat_model.py's generate() for the
    tokenizer-dependent half (looking up think_end_id, decoding each half)."""

    if think_end_id is not None and think_end_id in token_ids:
        split_at = token_ids.index(think_end_id) + 1
        return token_ids[:split_at], token_ids[split_at:]
    return [], token_ids


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

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int,
        config: GenerationConfig | None = None,
        steering: SteeringConfig | None = None,
        enable_thinking: bool | None = None,
        return_thinking: bool = False,
    ) -> str | tuple[str, str]:
        """enable_thinking overrides self.enable_thinking for this call only
        (None = use the instance default) -- needed when one ChatModel
        instance plays two roles that must not share a thinking-mode
        setting, e.g. self-judging in adversarial_screening.py: the agent
        may be generating with thinking on while safety_judge.py explicitly
        pins the same instance's judge calls to enable_thinking=False (a
        terse 1-5 digit judge response has no token budget for a reasoning
        block, and the judge is meant to be a fixed measurement instrument
        independent of the thing being manipulated on the agent side).

        return_thinking=True returns (content, thinking_text) instead of
        just content -- thinking_text is "" whenever no </think> token was
        emitted (enable_thinking=False, or a non-Qwen3 tokenizer). Default
        False keeps every existing caller's return type (str) unchanged."""

        config = config or GenerationConfig()
        effective_enable_thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        prompt_text = _build_prompt_text(self.tokenizer, messages, effective_enable_thinking)

        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.device)

        torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

        hook_handle = self._register_steering_hook(steering) if steering is not None else None
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :].tolist()
        think_end_id = self.tokenizer.convert_tokens_to_ids(_THINK_END_TAG)
        if think_end_id == self.tokenizer.unk_token_id:
            think_end_id = None
        thinking_ids, content_ids = _split_at_think_end(new_tokens, think_end_id)
        content_text = self.tokenizer.decode(content_ids, skip_special_tokens=True).strip()

        if not return_thinking:
            return content_text
        thinking_text = self.tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
        return content_text, thinking_text

    def next_token_logits(self, messages: list[dict[str, str]], enable_thinking: bool | None = None) -> np.ndarray:
        """One forward pass, no generation: the next-token logits at the
        generation position, shape (vocab_size,), float32.

        This is the deterministic-forward counterpart to `.generate()`'s
        first sampled/greedy token -- used by sycophancy_judge.py to turn the
        judge's greedy label into a full label-token distribution without
        regenerating anything. Unlike `generate()`, there is no RNG to seed
        here: a single forward pass over a fixed prompt is a pure function of
        the weights and the input, so (deliberately, unlike generate()'s
        per-call `torch.manual_seed`) this takes no `seed` argument and there
        is nothing to reseed."""

        effective_enable_thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        prompt_text = _build_prompt_text(self.tokenizer, messages, effective_enable_thinking)
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # bf16 tensors can't go straight to numpy; cast to float32 first.
        return outputs.logits[0, -1, :].float().cpu().numpy()

    def hidden_state_at_layer(self, messages: list[dict[str, str]], layer: int) -> np.ndarray:
        """One forward pass, no generation: returns the last-token residual
        stream activation at `layer` (transformers' `output_hidden_states=True`
        convention: hidden_states[0] is the embedding output, hidden_states[i]
        is the output of decoder layer i for i=1..num_layers). Used only for
        diff-in-means direction calibration (activation_direction.py) --
        generation-time steering uses _register_steering_hook instead, since
        a hook composes with `.generate()`'s incremental decoding while a
        single forward pass here does not."""

        prompt_text = _build_prompt_text(self.tokenizer, messages, self.enable_thinking)

        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        last_token_state = outputs.hidden_states[layer][0, -1, :]
        return last_token_state.float().cpu().numpy()

    def _register_steering_hook(self, steering: SteeringConfig):
        # Qwen3ForCausalLM (this project's only model family) exposes decoder
        # layers at .model.layers, standard for HF's Llama-style causal LMs.
        # A forward hook (rather than editing generate()'s internals) adds
        # the same delta at every decoding step, prefill included, and
        # composes with KV-cached incremental generation for free.
        # steering.layer - 1: see SteeringConfig's docstring for why this
        # offset is needed to land on the same tensor hidden_state_at_layer
        # read from.
        delta = torch.as_tensor(steering.direction, dtype=self.model.dtype, device=self.device) * steering.alpha
        target_layer = self.model.model.layers[steering.layer - 1]

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] + delta,) + output[1:]
            return output + delta

        return target_layer.register_forward_hook(hook)
