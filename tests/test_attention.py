"""Unit tests for attention variants."""

import pytest
import torch

from model import GPTConfig
from models.attention import (
    CausalSelfAttention,
    GroupedQueryAttention,
    RotaryEmbedding,
    apply_rotary_emb,
)


@pytest.fixture
def std_cfg():
    return GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dropout=0.0)


class TestRotaryEmbedding:
    def test_cos_sin_shapes(self, std_cfg):
        head_dim = std_cfg.n_embd // std_cfg.n_head
        rope = RotaryEmbedding(head_dim, max_seq_len=std_cfg.block_size, theta=std_cfg.rope_theta)
        dummy = torch.zeros(1)
        cos, sin = rope(dummy, seq_len=16)
        assert cos.shape[-1] == head_dim
        assert cos.shape[-2] == 16

    def test_cos_values_bounded(self, std_cfg):
        head_dim = std_cfg.n_embd // std_cfg.n_head
        rope = RotaryEmbedding(head_dim, max_seq_len=std_cfg.block_size)
        dummy = torch.zeros(1)
        cos, sin = rope(dummy, seq_len=8)
        assert cos.abs().max().item() <= 1.0 + 1e-5
        assert sin.abs().max().item() <= 1.0 + 1e-5

    def test_cos_sin_different(self, std_cfg):
        head_dim = std_cfg.n_embd // std_cfg.n_head
        rope = RotaryEmbedding(head_dim, max_seq_len=std_cfg.block_size)
        dummy = torch.zeros(1)
        cos, sin = rope(dummy, seq_len=16)
        assert not torch.allclose(cos, sin)


class TestApplyRotaryEmb:
    def test_output_shape_preserved(self, std_cfg):
        B, n_head, T = 2, std_cfg.n_head, 16
        head_dim = std_cfg.n_embd // n_head
        q = torch.randn(B, n_head, T, head_dim)
        k = torch.randn(B, n_head, T, head_dim)
        rope = RotaryEmbedding(head_dim, max_seq_len=std_cfg.block_size)
        cos, sin = rope(q, seq_len=T)
        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotation_preserves_norm(self, std_cfg):
        B, n_head, T = 1, std_cfg.n_head, 8
        head_dim = std_cfg.n_embd // n_head
        torch.manual_seed(7)
        q = torch.randn(B, n_head, T, head_dim)
        k = torch.randn(B, n_head, T, head_dim)
        rope = RotaryEmbedding(head_dim, max_seq_len=std_cfg.block_size)
        cos, sin = rope(q, seq_len=T)
        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


class TestCausalSelfAttention:
    def test_output_shape(self, std_cfg):
        attn = CausalSelfAttention(std_cfg)
        x = torch.randn(2, 16, std_cfg.n_embd)
        out = attn(x)
        assert out.shape == x.shape

    def test_single_token(self, std_cfg):
        attn = CausalSelfAttention(std_cfg)
        x = torch.randn(1, 1, std_cfg.n_embd)
        out = attn(x)
        assert out.shape == (1, 1, std_cfg.n_embd)

    def test_output_not_nan(self, std_cfg):
        torch.manual_seed(42)
        attn = CausalSelfAttention(std_cfg)
        x = torch.randn(2, 16, std_cfg.n_embd)
        out = attn(x)
        assert not torch.isnan(out).any()

    def test_causal_masking(self, std_cfg):
        torch.manual_seed(0)
        attn = CausalSelfAttention(std_cfg)
        attn.eval()
        x = torch.randn(1, 8, std_cfg.n_embd)
        out_full = attn(x)
        x_corrupt = x.clone()
        x_corrupt[:, 1:, :] = torch.randn_like(x_corrupt[:, 1:, :]) * 10
        out_corrupt = attn(x_corrupt)
        assert torch.allclose(out_full[:, 0, :], out_corrupt[:, 0, :], atol=1e-5)

    def test_rope_output_shape(self, std_cfg):
        std_cfg.pos_embedding = "rope"
        attn = CausalSelfAttention(std_cfg)
        x = torch.randn(2, 16, std_cfg.n_embd)
        out = attn(x)
        assert out.shape == x.shape

    def test_gradient_flows(self, std_cfg):
        attn = CausalSelfAttention(std_cfg)
        x = torch.randn(2, 8, std_cfg.n_embd, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


class TestGroupedQueryAttention:
    def _make_gqa(self, n_head=4, n_kv_heads=2, n_embd=64, block_size=32):
        cfg = GPTConfig(
            n_layer=2, n_head=n_head, n_kv_heads=n_kv_heads,
            n_embd=n_embd, block_size=block_size, vocab_size=100,
            dropout=0.0, attention_type="gqa",
        )
        return GroupedQueryAttention(cfg), cfg

    def test_output_shape_standard(self):
        attn, cfg = self._make_gqa(n_head=4, n_kv_heads=2)
        x = torch.randn(2, 16, cfg.n_embd)
        out = attn(x)
        assert out.shape == x.shape

    def test_mqa(self):
        attn, cfg = self._make_gqa(n_head=4, n_kv_heads=1)
        x = torch.randn(2, 16, cfg.n_embd)
        out = attn(x)
        assert out.shape == x.shape

    def test_mha_equivalent(self):
        attn, cfg = self._make_gqa(n_head=4, n_kv_heads=4)
        x = torch.randn(2, 16, cfg.n_embd)
        out = attn(x)
        assert out.shape == x.shape

    def test_fewer_kv_params(self):
        attn_gqa, _ = self._make_gqa(n_head=4, n_kv_heads=1)
        attn_mha, _ = self._make_gqa(n_head=4, n_kv_heads=4)
        n_gqa = sum(p.numel() for p in attn_gqa.parameters())
        n_mha = sum(p.numel() for p in attn_mha.parameters())
        assert n_gqa < n_mha

    def test_gradient_flows(self):
        attn, cfg = self._make_gqa(n_head=4, n_kv_heads=2)
        x = torch.randn(2, 8, cfg.n_embd, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_causal_masking_gqa(self):
        torch.manual_seed(1)
        attn, cfg = self._make_gqa(n_head=4, n_kv_heads=2)
        attn.eval()
        x = torch.randn(1, 8, cfg.n_embd)
        out_full = attn(x)
        x_corrupt = x.clone()
        x_corrupt[:, 1:, :] += 1000.0
        out_corrupt = attn(x_corrupt)
        assert torch.allclose(out_full[:, 0, :], out_corrupt[:, 0, :], atol=1e-4)
