"""
Attention variants: Standard, GQA, Flash Attention, RoPE

References:
- GQA: "GQA: Training Generalist Models" (Ainslie et al., 2023)
- Flash Attention 2: (Dao et al., 2023)
- RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.extensions import LoRALinear

# Detect PyTorch 2.0+ scaled_dot_product_attention (Flash Attention fallback).
_FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Pre-computes and caches cos/sin rotation matrices up to ``max_seq_len``.
    The matrices are registered as non-persistent buffers so they are moved to
    the correct device with the model but not saved in checkpoints.

    Reference: Su et al. (2021) https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Inverse frequencies: shape (dim // 2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute and cache the rotation matrices.
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)   # (T, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) matrices for the first ``seq_len`` positions.

        Extends the cache on-the-fly if ``seq_len`` exceeds the pre-computed
        length.

        Args:
            x: any tensor on the target device (used only for device/dtype).
            seq_len: length of the current sequence.

        Returns:
            cos: (1, 1, seq_len, dim)
            sin: (1, 1, seq_len, dim)
        """
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(x.dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension by –90°."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        q: (B, n_head, T, head_dim)
        k: (B, n_head, T, head_dim)  [or (B, n_kv_heads, T, head_dim) for GQA]
        cos: (1, 1, T, head_dim)
        sin: (1, 1, T, head_dim)

    Returns:
        q_rot, k_rot: same shapes as inputs.
    """
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention.

    Features:
    - Combined QKV projection for efficiency.
    - Causal (autoregressive) masking.
    - Flash Attention via ``torch.nn.functional.scaled_dot_product_attention``
      when available (PyTorch >= 2.0), with a manual fallback otherwise.
    - Optional Rotary Position Embeddings (RoPE).
    - Optional LoRA adaptation of c_attn and c_proj.
    """

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.pos_embedding = config.pos_embedding

        _Linear = LoRALinear if getattr(config, "use_lora", False) else nn.Linear

        if _Linear is LoRALinear:
            lora_kwargs = dict(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                bias=config.bias,
            )
            self.c_attn = LoRALinear(config.n_embd, 3 * config.n_embd, **lora_kwargs)
            self.c_proj = LoRALinear(config.n_embd, config.n_embd, **lora_kwargs)
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE
        if config.pos_embedding == "rope":
            self.rotary = RotaryEmbedding(
                self.head_dim,
                max_seq_len=config.block_size,
                theta=getattr(config, "rope_theta", 10000.0),
            )
        else:
            self.rotary = None

        # Causal mask (only used in the manual fallback path).
        if not _FLASH_AVAILABLE:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # QKV projection.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE if configured.
        if self.rotary is not None:
            cos, sin = self.rotary(x, T)
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Attention computation.
        if _FLASH_AVAILABLE:
            # scaled_dot_product_attention handles causal masking internally.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual implementation.
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale                # (B, nh, T, T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v                                             # (B, nh, T, hd)

        # Reassemble heads and project.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA).

    Uses ``n_head`` query heads but only ``n_kv_heads`` key/value heads
    (n_kv_heads <= n_head, n_head % n_kv_heads == 0).  KV heads are repeated
    to match the number of query heads before the dot-product.

    Reduces the KV cache size by a factor of ``n_head / n_kv_heads``, which
    lowers memory bandwidth pressure during inference.

    Also supports RoPE position embeddings.

    Reference: Ainslie et al. (2023) https://arxiv.org/abs/2305.13245
    """

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )
        n_kv_heads: int = config.n_kv_heads if config.n_kv_heads is not None else config.n_head
        assert config.n_head % n_kv_heads == 0, (
            f"n_head ({config.n_head}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.n_head = config.n_head
        self.n_kv_heads = n_kv_heads
        self.n_rep = config.n_head // n_kv_heads  # repeat factor
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.pos_embedding = config.pos_embedding

        # Separate Q and KV projections.
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.kv_proj = nn.Linear(config.n_embd, 2 * n_kv_heads * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE
        if config.pos_embedding == "rope":
            self.rotary = RotaryEmbedding(
                self.head_dim,
                max_seq_len=config.block_size,
                theta=getattr(config, "rope_theta", 10000.0),
            )
        else:
            self.rotary = None

        # Causal mask fallback.
        if not _FLASH_AVAILABLE:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
                persistent=False,
            )

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Expand KV heads to match the number of query heads.

        Args:
            x: (B, n_kv_heads, T, head_dim)
            n_rep: repetition factor

        Returns:
            (B, n_kv_heads * n_rep, T, head_dim)
        """
        if n_rep == 1:
            return x
        B, n_kv_heads, T, hd = x.shape
        return (
            x[:, :, None, :, :]
            .expand(B, n_kv_heads, n_rep, T, hd)
            .reshape(B, n_kv_heads * n_rep, T, hd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V.
        q = self.q_proj(x)    # (B, T, n_head * head_dim)
        kv = self.kv_proj(x)  # (B, T, 2 * n_kv_heads * head_dim)
        k, v = kv.split(self.n_kv_heads * self.head_dim, dim=-1)

        # Reshape.
        q = q.view(B, T, self.n_head,    self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, nkv, T, hd)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, nkv, T, hd)

        # Apply RoPE.
        if self.rotary is not None:
            cos, sin = self.rotary(x, T)
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Expand KV to match Q heads.
        k = self._repeat_kv(k, self.n_rep)  # (B, nh, T, hd)
        v = self._repeat_kv(v, self.n_rep)  # (B, nh, T, hd)

        # Attention.
        if _FLASH_AVAILABLE:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
