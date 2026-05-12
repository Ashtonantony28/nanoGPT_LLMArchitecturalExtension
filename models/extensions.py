"""
Extension modules: RMSNorm, SwiGLU, LoRA, MoE

References:
- RMSNorm: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
- SwiGLU: "GLU Variants Improve Transformer" (Shazeer et al., 2020)
- LoRA: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- MoE: "Switch Transformers" (Fedus et al., 2022)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm omits the mean-centering step and learns only a
    gain parameter, which reduces computation while retaining re-scaling
    invariance.  Used in LLaMA, Mistral, and many modern LLMs.

    Reference: Zhang & Sennrich (2019) https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS over the last dimension.  Keep dtype for numerical
        # stability but compute the norm in float32.
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(dtype) * self.weight


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network.

    Replaces the standard ``GELU(xW_1) * W_2`` MLP with a gated variant:
        output = (SiLU(gate_proj(x)) * up_proj(x)) @ down_proj

    The hidden dimension is scaled by 2/3 * 4 * n_embd so that the parameter
    count roughly matches a standard 4x MLP.  Rounded to the nearest multiple
    of 256 for hardware efficiency.

    Reference: Shazeer (2020) https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config) -> None:
        super().__init__()
        # Match parameter count to a 4x GELU MLP: two matrices vs. one.
        hidden_dim = int(2 * 4 * config.n_embd / 3)
        # Round up to nearest multiple of 256 for hardware efficiency.
        hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: element-wise product of gated branch and up-projection.
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class LoRALinear(nn.Module):
    """LoRA-adapted linear layer.

    Adds a low-rank decomposition ``BA`` alongside a frozen base weight ``W``:
        output = x @ W^T + x @ A^T @ B^T * scaling

    During training only A and B are updated; W is kept frozen.  At inference
    the delta can be merged into W to eliminate the extra computation.

    Reference: Hu et al. (2021) https://arxiv.org/abs/2106.09685
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.merged = False

        # Base (frozen) weight.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Low-rank matrices.  A uses kaiming uniform; B is initialised to zero
        # so the LoRA delta is zero at the start of training.
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B is already zeros.

    def merge(self) -> None:
        """Merge LoRA delta into the base weight for efficient inference."""
        if self.merged:
            return
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.merged = True

    def unmerge(self) -> None:
        """Unmerge LoRA delta from the base weight (e.g. before fine-tuning)."""
        if not self.merged:
            return
        self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if self.merged:
            return base
        # Add the low-rank delta.
        lora_delta = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),
            self.lora_B,
        ) * self.scaling
        return base + lora_delta


# ---------------------------------------------------------------------------
# Mixture of Experts
# ---------------------------------------------------------------------------

class _ExpertMLP(nn.Module):
    """Single expert: a two-layer MLP with GELU activation."""

    def __init__(self, n_embd: int, bias: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class MoEBlock(nn.Module):
    """Mixture-of-Experts feed-forward block with top-k routing.

    Each token is independently routed to ``num_experts_per_tok`` experts.
    The router is a simple linear layer whose softmax weights are used both
    to select top-k experts and to compute a weighted sum of their outputs.

    A load-balancing auxiliary loss (from Switch Transformers) encourages
    uniform expert utilisation:
        aux_loss = num_experts * sum_i(f_i * p_i)
    where f_i is the fraction of tokens dispatched to expert i and p_i is the
    mean router probability for expert i.

    Reference: Fedus et al. (2022) https://arxiv.org/abs/2101.03961
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_embd = config.n_embd

        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_ExpertMLP(config.n_embd, bias=config.bias) for _ in range(config.num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C) input tensor.

        Returns:
            output: (B, T, C) MoE output.
            aux_loss: scalar load-balancing loss.
        """
        B, T, C = x.shape
        # Flatten to (B*T, C) for routing.
        x_flat = x.view(-1, C)  # (S, C) where S = B*T

        # Router logits and probabilities.
        router_logits = self.router(x_flat)          # (S, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (S, E)

        # Select top-k experts per token.
        topk_probs, topk_indices = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)
        # Renormalise top-k weights so they sum to 1.
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # (S, K)

        # Compute output by dispatching each token to its selected experts.
        output = torch.zeros_like(x_flat)  # (S, C)
        for k in range(self.num_experts_per_tok):
            expert_idx = topk_indices[:, k]   # (S,)
            weight = topk_probs[:, k]         # (S,)
            for e in range(self.num_experts):
                # Mask of tokens routed to expert e at position k.
                mask = expert_idx == e          # (S,) bool
                if not mask.any():
                    continue
                expert_out = self.experts[e](x_flat[mask])  # (n, C)
                output[mask] += weight[mask].unsqueeze(-1) * expert_out

        output = output.view(B, T, C)

        # Load-balancing auxiliary loss.
        # f_i: fraction of tokens dispatched to each expert.
        # p_i: mean router probability for each expert.
        S = B * T
        # One-hot indicator of which tokens were dispatched to each expert
        # (across all top-k slots).
        dispatch_mask = torch.zeros(S, self.num_experts, device=x.device, dtype=x.dtype)
        for k in range(self.num_experts_per_tok):
            dispatch_mask.scatter_add_(
                1,
                topk_indices[:, k].unsqueeze(1),
                torch.ones(S, 1, device=x.device, dtype=x.dtype),
            )
        # Clamp so each expert is counted at most once per token.
        dispatch_mask = dispatch_mask.clamp(max=1.0)

        f = dispatch_mask.mean(dim=0)           # (E,) fraction of tokens per expert
        p = router_probs.mean(dim=0)            # (E,) mean router prob per expert
        aux_loss = self.num_experts * (f * p).sum()

        return output, aux_loss
