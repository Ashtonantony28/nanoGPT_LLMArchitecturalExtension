"""Unit tests for extension modules: RMSNorm, SwiGLU, LoRA, MoE."""

import pytest
import torch
import math

from model import GPTConfig
from models.extensions import RMSNorm, SwiGLUMLP, LoRALinear, MoEBlock


@pytest.fixture
def tiny_cfg():
    return GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=100, dropout=0.0)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalizes_rms(self):
        norm = RMSNorm(64, eps=0.0)
        norm.weight.data.fill_(1.0)
        x = torch.randn(4, 8, 64) * 5.0
        out = norm(x)
        # RMS of each token embedding should be close to 1
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_learnable_weight(self):
        norm = RMSNorm(32)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (32,)

    def test_no_bias_parameter(self):
        norm = RMSNorm(32)
        assert not hasattr(norm, "bias") or norm.bias is None

    def test_gradient_flows(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert norm.weight.grad is not None

    def test_different_dims(self):
        for dim in [16, 32, 64, 128, 256]:
            norm = RMSNorm(dim)
            x = torch.randn(1, 4, dim)
            out = norm(x)
            assert out.shape == (1, 4, dim)


# ---------------------------------------------------------------------------
# SwiGLUMLP
# ---------------------------------------------------------------------------

class TestSwiGLUMLP:
    def test_output_shape(self, tiny_cfg):
        mlp = SwiGLUMLP(tiny_cfg)
        x = torch.randn(2, 16, tiny_cfg.n_embd)
        out = mlp(x)
        assert out.shape == x.shape

    def test_output_not_nan(self, tiny_cfg):
        mlp = SwiGLUMLP(tiny_cfg)
        x = torch.randn(2, 8, tiny_cfg.n_embd)
        out = mlp(x)
        assert not torch.isnan(out).any()

    def test_batch_size_one(self, tiny_cfg):
        mlp = SwiGLUMLP(tiny_cfg)
        x = torch.randn(1, 4, tiny_cfg.n_embd)
        out = mlp(x)
        assert out.shape == x.shape

    def test_gradient_flows(self, tiny_cfg):
        mlp = SwiGLUMLP(tiny_cfg)
        x = torch.randn(2, 8, tiny_cfg.n_embd, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None

    def test_different_from_gelu_mlp(self, tiny_cfg):
        """SwiGLU output should differ from a standard GELU MLP."""
        torch.manual_seed(0)
        swiglu = SwiGLUMLP(tiny_cfg)
        x = torch.randn(2, 8, tiny_cfg.n_embd)
        out1 = swiglu(x)
        out2 = swiglu(x)
        assert torch.allclose(out1, out2)  # deterministic


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

class TestLoRALinear:
    def test_output_shape(self):
        layer = LoRALinear(64, 128, rank=4, alpha=8.0)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == (2, 16, 128)

    def test_square_output_shape(self):
        layer = LoRALinear(64, 64, rank=8, alpha=16.0)
        x = torch.randn(3, 8, 64)
        out = layer(x)
        assert out.shape == (3, 8, 64)

    def test_lora_b_initialized_zero(self):
        layer = LoRALinear(64, 32, rank=4, alpha=8.0)
        assert torch.allclose(layer.lora_B, torch.zeros_like(layer.lora_B))

    def test_lora_a_not_all_zero(self):
        torch.manual_seed(42)
        layer = LoRALinear(64, 32, rank=4, alpha=8.0)
        assert layer.lora_A.abs().sum().item() > 0.0

    def test_scaling_correct(self):
        rank, alpha = 4, 16.0
        layer = LoRALinear(32, 32, rank=rank, alpha=alpha)
        assert abs(layer.scaling - alpha / rank) < 1e-6

    def test_gradient_flows(self):
        layer = LoRALinear(64, 32, rank=4, alpha=8.0)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert layer.lora_A.grad is not None

    def test_lora_b_zero_init_means_identity_like_output(self):
        """With lora_B=0 the LoRA delta is zero; output equals base linear."""
        torch.manual_seed(0)
        layer = LoRALinear(32, 32, rank=4, alpha=8.0, bias=False)
        x = torch.randn(1, 4, 32)
        # Manually confirm lora_B is zero
        assert layer.lora_B.abs().max().item() == 0.0
        out_lora = layer(x)
        import torch.nn.functional as F
        out_base = F.linear(x, layer.weight)
        assert torch.allclose(out_lora, out_base, atol=1e-6)

    def test_merge_and_unmerge(self):
        torch.manual_seed(5)
        layer = LoRALinear(32, 16, rank=4, alpha=8.0, bias=False)
        # Give lora_B non-zero values so merge has an effect
        with torch.no_grad():
            layer.lora_B.fill_(0.1)
        x = torch.randn(1, 4, 32)
        out_before = layer(x).detach().clone()
        layer.merge()
        assert layer.merged
        out_merged = layer(x).detach().clone()
        assert torch.allclose(out_before, out_merged, atol=1e-5)
        layer.unmerge()
        assert not layer.merged
        out_unmerged = layer(x).detach().clone()
        assert torch.allclose(out_before, out_unmerged, atol=1e-5)

    def test_with_bias(self):
        layer = LoRALinear(32, 16, rank=4, alpha=8.0, bias=True)
        assert layer.bias is not None
        x = torch.randn(1, 4, 32)
        out = layer(x)
        assert out.shape == (1, 4, 16)

    def test_without_bias(self):
        layer = LoRALinear(32, 16, rank=4, alpha=8.0, bias=False)
        assert layer.bias is None
        x = torch.randn(1, 4, 32)
        out = layer(x)
        assert out.shape == (1, 4, 16)


# ---------------------------------------------------------------------------
# MoEBlock
# ---------------------------------------------------------------------------

class TestMoEBlock:
    def _make_moe_cfg(self, num_experts=4, num_experts_per_tok=2):
        return GPTConfig(
            n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=100,
            dropout=0.0, use_moe=True,
            num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
        )

    def test_output_shape(self):
        cfg = self._make_moe_cfg()
        moe = MoEBlock(cfg)
        x = torch.randn(2, 16, cfg.n_embd)
        out, aux = moe(x)
        assert out.shape == x.shape

    def test_aux_loss_is_scalar(self):
        cfg = self._make_moe_cfg()
        moe = MoEBlock(cfg)
        x = torch.randn(2, 8, cfg.n_embd)
        _, aux = moe(x)
        assert aux.ndim == 0 or aux.numel() == 1

    def test_aux_loss_is_nonnegative(self):
        cfg = self._make_moe_cfg()
        moe = MoEBlock(cfg)
        x = torch.randn(2, 8, cfg.n_embd)
        _, aux = moe(x)
        assert aux.item() >= 0.0

    def test_output_not_nan(self):
        torch.manual_seed(42)
        cfg = self._make_moe_cfg()
        moe = MoEBlock(cfg)
        x = torch.randn(2, 8, cfg.n_embd)
        out, _ = moe(x)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self):
        cfg = self._make_moe_cfg()
        moe = MoEBlock(cfg)
        x = torch.randn(2, 8, cfg.n_embd, requires_grad=True)
        out, aux = moe(x)
        (out.sum() + aux).backward()
        assert x.grad is not None

    def test_mqa_top1(self):
        cfg = self._make_moe_cfg(num_experts=4, num_experts_per_tok=1)
        moe = MoEBlock(cfg)
        x = torch.randn(2, 8, cfg.n_embd)
        out, aux = moe(x)
        assert out.shape == x.shape

    def test_many_experts(self):
        cfg = self._make_moe_cfg(num_experts=8, num_experts_per_tok=2)
        moe = MoEBlock(cfg)
        x = torch.randn(2, 8, cfg.n_embd)
        out, aux = moe(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# LoRA round-trip via GPT model
# ---------------------------------------------------------------------------

class TestLoRAStateDictViaGPT:
    def test_lora_keys_in_state_dict(self, tiny_cfg):
        from model import GPT
        tiny_cfg.use_lora = True
        tiny_cfg.lora_rank = 4
        tiny_cfg.lora_alpha = 8.0
        model = GPT(tiny_cfg)
        lora_sd = model.get_lora_state_dict()
        assert len(lora_sd) > 0
        for k in lora_sd:
            assert "lora_A" in k or "lora_B" in k

    def test_lora_load_roundtrip(self, tiny_cfg):
        from model import GPT
        tiny_cfg.use_lora = True
        tiny_cfg.lora_rank = 4
        tiny_cfg.lora_alpha = 8.0
        model = GPT(tiny_cfg)
        sd = model.get_lora_state_dict()
        # Modify values
        with torch.no_grad():
            for v in sd.values():
                v.fill_(0.5)
        model.load_lora_state_dict(sd)
        sd2 = model.get_lora_state_dict()
        for k in sd:
            assert torch.allclose(sd[k], sd2[k])
