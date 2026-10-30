import pytest
import torch
from torch import nn

from cartridges.core import TrainableKVCartridge
from cartridges.sweep.fake_quant import analytic_weight_bytes, fake_quantize_, quant_dequant
from cartridges.sweep.kv_quant import kv_bytes_per_slot, with_kv_precision
from cartridges.sweep.longhealth import eval_items, score


class _TiedLM(nn.Module):
    def __init__(self, vocab=512, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head


@pytest.mark.parametrize("bits,max_rel", [(8, 0.01), (4, 0.15)])
def test_quant_dequant_error(bits, max_rel):
    torch.manual_seed(0)
    w = torch.randn(64, 512)
    deq = quant_dequant(w, bits, 128)
    assert deq.dtype == w.dtype and deq.shape == w.shape
    assert ((deq - w).norm() / w.norm()).item() < max_rel


def test_quant_dequant_levels_and_zero():
    torch.manual_seed(0)
    w = torch.randn(4, 128)
    w[0, 5] = 0.0
    deq = quant_dequant(w, 4, 128)
    assert deq[0, 5].item() == 0.0
    assert all(len(torch.unique(row)) <= 16 for row in deq)


def test_quant_dequant_rejects_bad_group():
    with pytest.raises(ValueError):
        quant_dequant(torch.randn(4, 100), 4, 128)


def test_fake_quantize_includes_tied_embedding():
    torch.manual_seed(0)
    model = _TiedLM()
    before = model.embed.weight.detach().clone()
    norm_before = model.norm.weight.detach().clone()
    report = fake_quantize_(model, "int4", 128)
    assert model.lm_head.weight is model.embed.weight
    assert report["embedding_tied"]
    assert report["quantized_tensors"] == 2
    assert not torch.equal(model.embed.weight, before)
    assert torch.equal(model.norm.weight, norm_before)


def test_analytic_weight_bytes():
    model = _TiedLM(vocab=512, hidden=256)
    q_params = 512 * 256 + 256 * 256
    other = 2 * 256
    r4 = analytic_weight_bytes(model, "int4", 128)
    assert r4["quantized_params"] == q_params
    assert r4["embedding_params"] == 512 * 256
    assert r4["weight_bytes"] == int(q_params / 2 + (q_params // 128) * 4 + other * 2)
    assert analytic_weight_bytes(model, "bf16", 128)["weight_bytes"] == (q_params + other) * 2


@pytest.mark.parametrize("layers,heads,dim,expected", [
    (36, 8, 128, 147456),   # Qwen3-8B
    (28, 8, 128, 114688),   # Qwen3-0.6B
    (18, 1, 256, 18432),    # Gemma 3 270M
])
def test_kv_bytes_per_slot(layers, heads, dim, expected):
    assert kv_bytes_per_slot(layers, heads, dim, "bf16") == expected
    assert kv_bytes_per_slot(layers, heads, dim, "int8") == 2 * layers * heads * (dim + 2)


def test_int8_kv_roundtrip():
    torch.manual_seed(0)
    keys = [torch.randn(1, 2, 16, 64, dtype=torch.bfloat16) for _ in range(3)]
    values = [torch.randn(1, 2, 16, 64, dtype=torch.bfloat16) for _ in range(3)]
    cart = TrainableKVCartridge(keys=keys, values=values, num_frozen_tokens=1)
    q = with_kv_precision(cart, "int8")
    assert q.num_tokens == 16 and q.num_frozen_tokens == 1
    k0, _ = cart.layer(0)
    qk0, _ = q.layer(0)
    assert qk0.dtype == k0.dtype and qk0.shape == k0.shape
    assert ((qk0.float() - k0.float()).norm() / k0.float().norm()).item() < 0.02
    assert with_kv_precision(cart, "bf16") is cart


def _record():
    return {"questions": [
        {"question": "Which drug?", "answer_a": "Aspirin", "answer_b": "Heparin",
         "answer_c": "Insulin", "answer_d": "Morphine", "answer_e": "None",
         "correct": "Heparin"},
        {"question": "Odd one?", "answer_a": "x", "answer_b": "y", "answer_c": "z",
         "answer_d": "w", "answer_e": "v", "correct": "not an option"},
    ]}


def test_eval_items_and_score():
    items = eval_items(_record(), 1)
    assert items[0]["correct_letter"] == "B" and items[0]["id"] == "lh_p01_q00"
    assert "B) Heparin" in items[0]["prompt"]
    assert items[1]["correct_letter"] is None
    assert score("B) Heparin", items[0]) == {"substring_hit": True, "letter_hit": True}
    assert score("(B)", items[0]) == {"substring_hit": False, "letter_hit": True}
    assert score("The answer is heparin.", items[0]) == {"substring_hit": True, "letter_hit": False}
    assert score("Bleeding risk", items[0])["letter_hit"] is False
    assert score("The correct answer is: **B) Heparin**", items[0])["letter_hit"] is True
    assert score("**Correct Answer: A) Aspirin**\n**Incorrect: B)**", items[0])["letter_hit"] is False
    assert score("B", items[1])["letter_hit"] is False
