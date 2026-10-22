"""GPT-2 weight loading: name mapping checked against HuggingFace's implementation."""

import pytest
import torch

from model import GPT, GPTConfig

transformers = pytest.importorskip("transformers")


def make_hf_pair():
    torch.manual_seed(0)
    hf_cfg = transformers.GPT2Config(
        n_layer=2, n_head=2, n_embd=64, n_positions=32, vocab_size=100,
        activation_function="gelu", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    hf = transformers.GPT2LMHeadModel(hf_cfg).eval()
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=100, dropout=0.0)
    model = GPT(cfg).eval()
    model.load_hf_state_dict(hf.state_dict())
    return hf, model


def test_load_hf_state_dict_matches_hf_logits():
    hf, model = make_hf_pair()
    idx = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        ours, _ = model(idx)
        theirs = hf(idx).logits
    assert (ours - theirs).abs().max().item() < 1e-4


def test_load_hf_state_dict_keeps_lora_init_and_matches_hf_logits():
    hf, _ = make_hf_pair()
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=100,
                    dropout=0.0, use_lora=True, lora_rank=4, lora_alpha=8.0)
    model = GPT(cfg).eval()
    model.load_hf_state_dict(hf.state_dict())
    idx = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        ours, _ = model(idx)
        theirs = hf(idx).logits
    assert (ours - theirs).abs().max().item() < 1e-4


def test_from_pretrained_rejects_architecture_overrides():
    with pytest.raises(AssertionError):
        GPT.from_pretrained("gpt2", {"attention_type": "gqa"})


def test_from_pretrained_gpt2_logits_match_hf():
    try:
        hf = transformers.GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True).eval()
    except OSError:
        pytest.skip("gpt2 not in the HF cache; run once with network access to cache it")
    model = GPT.from_pretrained("gpt2").eval()
    idx = torch.randint(0, 50257, (1, 32), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ours, _ = model(idx)
        theirs = hf(idx).logits
    assert (ours - theirs).abs().max().item() < 1e-4
