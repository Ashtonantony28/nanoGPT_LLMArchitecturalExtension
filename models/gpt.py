"""
Extended GPT model composition with all SOTA extensions.

Provides:
- MLP: standard GELU feedforward (with optional LoRA)
- Block: transformer block supporting LayerNorm/RMSNorm, standard/SwiGLU MLP,
  standard/GQA attention, and MoE feedforward layers.

Both MLP and Block are thin "glue" modules that delegate to the feature
implementations in models/extensions.py and models/attention.py.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.extensions import LoRALinear, RMSNorm, SwiGLUMLP, MoEBlock
from models.attention import CausalSelfAttention, GroupedQueryAttention


# ---------------------------------------------------------------------------
# Standard MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Standard two-layer feed-forward network with GELU activation.

    Projects the input from ``n_embd`` to ``4 * n_embd``, applies GELU, then
    projects back.  Optionally uses LoRALinear for the linear layers when
    ``config.use_lora`` is True.

    GPT-2 naming convention is preserved (``c_fc`` / ``c_proj``) so that
    pretrained weights load correctly.
    """

    def __init__(self, config) -> None:
        super().__init__()

        if getattr(config, "use_lora", False):
            lora_kwargs = dict(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                bias=config.bias,
            )
            self.c_fc = LoRALinear(config.n_embd, 4 * config.n_embd, **lora_kwargs)
            self.c_proj = LoRALinear(4 * config.n_embd, config.n_embd, **lora_kwargs)
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Single transformer block (pre-LN architecture).

    Configuration-driven feature matrix
    ─────────────────────────────────────
    Normalisation  ``config.norm_type``    'layernorm' | 'rmsnorm'
    Attention      ``config.attention_type`` 'standard' | 'gqa'
    Feed-forward   ``config.activation``   'gelu' → MLP
                                           'swiglu' → SwiGLUMLP
    Mixture        ``config.use_moe``      True → MoEBlock (overrides activation)

    When ``use_moe=True`` the block's ``forward`` returns a ``(output, aux_loss)``
    tuple.  Otherwise it returns just the output tensor with ``aux_loss = 0.0``.
    This makes the calling convention uniform: callers should always unpack two
    values and accumulate ``aux_loss``.
    """

    def __init__(self, config) -> None:
        super().__init__()

        # Normalisation factory.
        def _make_norm() -> nn.Module:
            if getattr(config, "norm_type", "layernorm") == "rmsnorm":
                return RMSNorm(config.n_embd)
            return nn.LayerNorm(config.n_embd, bias=config.bias)

        self.ln_1 = _make_norm()
        self.ln_2 = _make_norm()

        # Attention.
        attn_type = getattr(config, "attention_type", "standard")
        if attn_type == "gqa":
            self.attn = GroupedQueryAttention(config)
        else:
            self.attn = CausalSelfAttention(config)

        # Feed-forward.
        self.use_moe: bool = getattr(config, "use_moe", False)
        if self.use_moe:
            self.ff = MoEBlock(config)
        elif getattr(config, "activation", "gelu") == "swiglu":
            self.ff = SwiGLUMLP(config)
        else:
            self.ff = MLP(config)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, T, C) input tensor.

        Returns:
            output: (B, T, C) transformed tensor.
            aux_loss: scalar auxiliary loss from MoE routing (0.0 when no MoE).
        """
        # Pre-LN self-attention with residual.
        x = x + self.attn(self.ln_1(x))

        # Pre-LN feed-forward with residual.
        aux_loss: torch.Tensor
        if self.use_moe:
            ff_out, aux_loss = self.ff(self.ln_2(x))
        else:
            ff_out = self.ff(self.ln_2(x))
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        x = x + ff_out
        return x, aux_loss
