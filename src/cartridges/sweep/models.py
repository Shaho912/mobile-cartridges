"""Model loading (with fake quantization) and one prompt path shared by teacher, training, eval."""

from __future__ import annotations

import re

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from cartridges.core import TrainableKVCartridge
from cartridges.sweep.fake_quant import fake_quantize_

_TURN_END_TOKENS = {"qwen3": "<|im_end|>", "gemma3": "<end_of_turn>"}


def model_family(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen3" in lowered:
        return "qwen3"
    if "gemma-3" in lowered:
        return "gemma3"
    raise ValueError(f"No prompt adapter for {model_id}; add one in cartridges.sweep.models.")


def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def text_config(config):
    """Gemma 3 multimodal checkpoints nest the LM config; text-only ones do not."""
    return getattr(config, "text_config", config)


def attention_info(model_id: str) -> dict:
    """Shape facts read from config.json: everything the budget math and fit checks need."""
    cfg = text_config(AutoConfig.from_pretrained(model_id))
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    layer_types = list(getattr(cfg, "layer_types", None) or [])
    sliding_window = getattr(cfg, "sliding_window", None)
    if not getattr(cfg, "use_sliding_window", True):
        sliding_window = None
    return {
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": head_dim,
        "max_position_embeddings": cfg.max_position_embeddings,
        "sliding_window": sliding_window,
        "num_sliding_layers": sum(t == "sliding_attention" for t in layer_types),
        "num_global_layers": sum(t == "full_attention" for t in layer_types) or cfg.num_hidden_layers,
    }


def _enforce_tied_embeddings(model) -> None:
    """Keep one shared embedding/lm_head when the config says they are tied.

    Some transformers versions load a checkpoint that stores both tensors as two separate
    copies, which would double-count ~155M params (Qwen3-0.6B) in the budget and quantize
    them separately. Re-tie only if the two copies are identical.
    """
    if not getattr(text_config(model.config), "tie_word_embeddings", False):
        return
    inp, out = model.get_input_embeddings(), model.get_output_embeddings()
    if out is None or out.weight is inp.weight:
        return
    if not torch.equal(out.weight, inp.weight):
        raise RuntimeError("Config says tie_word_embeddings but lm_head differs from the embedding.")
    out.weight = inp.weight


def load_model(model_id: str, quant: str, group_size: int, device: str):
    """Load in bf16, then fake-quantize every Linear/Embedding weight (embedding included)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    _enforce_tied_embeddings(model)
    report = fake_quantize_(model, quant, group_size)
    return model, tokenizer, report


class PromptAdapter:
    """Chat formatting that is identical for the teacher, cartridge training, and eval.

    Thinking is disabled through the template (Qwen3 ``enable_thinking=False``) rather than
    a ``/no_think`` string, so no model sees Qwen-specific text. When the template starts with
    BOS (Gemma), BOS goes at cartridge position 0 and is stripped from the query that follows.
    """

    def __init__(self, tokenizer, model_id: str):
        self.tokenizer = tokenizer
        self.family = model_family(model_id)
        self.template_kwargs = {"enable_thinking": False} if self.family == "qwen3" else {}
        self.turn_end_id = tokenizer.convert_tokens_to_ids(_TURN_END_TOKENS[self.family])
        self.stop_ids = {self.turn_end_id}
        if tokenizer.eos_token_id is not None:
            self.stop_ids.add(tokenizer.eos_token_id)
        probe = self.chat_ids([{"role": "user", "content": "x"}])
        bos = tokenizer.bos_token_id
        self.template_bos = bos is not None and probe[0] == bos

    def chat_ids(self, messages: list[dict[str, str]]) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def query_ids(self, tokenizer, user_message: str) -> torch.Tensor:
        """Student prompt placed after a cartridge. Signature matches ``train_cartridge``."""
        del tokenizer
        ids = self.chat_ids([{"role": "user", "content": user_message}])
        if self.template_bos:
            ids = ids[1:]
        return torch.tensor([ids], dtype=torch.long)

    def init_text(self, system_prompt: str) -> str:
        return (self.tokenizer.bos_token + system_prompt) if self.template_bos else system_prompt

    def num_text_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


@torch.no_grad()
def greedy_generate(model, input_ids: torch.Tensor, cache, stop_ids: set[int], max_new_tokens: int):
    """Greedy decode; ``cache`` is a fresh DynamicCache (cartridge) or None."""
    outputs = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    past = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        if token_id in stop_ids:
            break
        generated.append(token_id)
        outputs = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    return generated


def clean_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.replace("</think>", "").strip()


def cache_visibility(model, cartridge: TrainableKVCartridge) -> dict:
    """How many cartridge slots each layer actually keeps once wrapped as a model cache.

    Sliding-window layers (Gemma 3 local layers) may keep only the most recent window of
    slots. The reported sequence length must still equal the slot count, otherwise query
    positions (RoPE) would be misplaced, and that is raised as an error.
    """
    cache = cartridge.as_cache(model.config)
    stored = [int(layer.keys.shape[-2]) for layer in cache.layers]
    seq_len = int(cache.get_seq_length())
    if seq_len != cartridge.num_tokens:
        raise RuntimeError(
            f"Cache reports seq length {seq_len} for a {cartridge.num_tokens}-slot cartridge; "
            "query positions would not follow the cartridge."
        )
    return {
        "slots": cartridge.num_tokens,
        "stored_slots_per_layer": stored,
        "layers_seeing_all_slots": sum(s == cartridge.num_tokens for s in stored),
        "min_stored_slots": min(stored),
    }
