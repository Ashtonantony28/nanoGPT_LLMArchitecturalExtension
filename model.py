"""
Extended GPT language model with SOTA extensions.

Builds on the nanoGPT codebase (Karpathy, 2022) with the following additions:
  - RMSNorm / SwiGLU / LoRA / MoE via models/extensions.py
  - Grouped-Query Attention and RoPE via models/attention.py
  - Gradient checkpointing for large-model training
  - Top-p (nucleus) sampling alongside top-k
  - GPT-2 pretrained weight loading from HuggingFace Transformers
  - LoRA state-dict helpers for parameter-efficient fine-tuning
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from models.extensions import RMSNorm
from models.attention import RotaryEmbedding
from models.gpt import MLP, Block


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """All hyper-parameters for the GPT model.

    Default values match GPT-2 small (124 M parameters).
    """

    # Core architecture
    block_size: int = 1024
    vocab_size: int = 50304          # GPT-2 vocab rounded up to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True                # use bias in Linear and LayerNorm layers

    # Attention variant
    attention_type: str = "standard" # 'standard' | 'gqa'
    n_kv_heads: Optional[int] = None # GQA key/value heads; None → same as n_head

    # Position embedding
    pos_embedding: str = "learned"   # 'learned' | 'rope' | 'none'
    rope_theta: float = 10000.0      # RoPE base frequency

    # Normalisation
    norm_type: str = "layernorm"     # 'layernorm' | 'rmsnorm'

    # Feed-forward activation
    activation: str = "gelu"        # 'gelu' | 'swiglu'

    # LoRA
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    # Mixture of Experts
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_aux_coef: float = 0.01       # weight on the load-balancing loss (not applied yet, see SG6)

    # Memory / training efficiency
    gradient_checkpointing: bool = False
    checkpoint_every_n_layers: int = 1   # checkpoint every N-th layer


# ---------------------------------------------------------------------------
# GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """Extended GPT language model.

    The model follows the original GPT-2 architecture (decoder-only transformer
    with pre-LN) but supports a rich set of modern extensions controlled by
    :class:`GPTConfig`.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # ── Embeddings ──────────────────────────────────────────────────────
        wte = nn.Embedding(config.vocab_size, config.n_embd)

        if config.pos_embedding == "learned":
            wpe: Optional[nn.Embedding] = nn.Embedding(config.block_size, config.n_embd)
        else:
            wpe = None  # RoPE is handled inside each attention layer

        # ── Transformer blocks ───────────────────────────────────────────────
        blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # ── Final normalisation ──────────────────────────────────────────────
        if config.norm_type == "rmsnorm":
            ln_f: nn.Module = RMSNorm(config.n_embd)
        else:
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

        self.transformer = nn.ModuleDict(
            {
                "wte": wte,
                **({"wpe": wpe} if wpe is not None else {}),
                "drop": nn.Dropout(config.dropout),
                "h": blocks,
                "ln_f": ln_f,
            }
        )

        # ── Language model head (weight-tied to wte) ─────────────────────────
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: share the embedding matrix with the output projection.
        # This is the GPT-2 approach and saves ~38 M params for the 124 M model.
        self.lm_head.weight = self.transformer["wte"].weight

        # Initialise all weights.
        self.apply(self._init_weights)

        # Apply scaled initialisation to residual projections (GPT-2 paper §2.3).
        # c_proj in attention and MLP are the "residual path" projections.
        for pname, p in self.named_parameters():
            if pname.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report parameter count.
        n_params = self.get_num_params()
        print(f"GPT model initialised: {n_params / 1e6:.2f} M parameters")

    # ── Weight initialisation ────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 weight initialisation scheme."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)

    # ── Parameter utilities ──────────────────────────────────────────────────

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of trainable parameters.

        Args:
            non_embedding: when True, subtract the position-embedding parameters
                (they don't contribute to the model's expressive capacity in the
                same way as weight matrices).  Token embeddings are still counted
                because they are weight-tied to the lm_head.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and "wpe" in self.transformer:
            n_params -= self.transformer["wpe"].weight.numel()
        return n_params

    def crop_block_size(self, block_size: int) -> None:
        """Crop the model to support shorter sequence lengths.

        Useful when fine-tuning a pretrained model on shorter contexts:
        the position embedding table is truncated in-place.

        Args:
            block_size: new (smaller) maximum sequence length.
        """
        assert block_size <= self.config.block_size, (
            f"Cannot crop to {block_size} > current block_size {self.config.block_size}"
        )
        self.config.block_size = block_size
        if "wpe" in self.transformer:
            self.transformer["wpe"].weight = nn.Parameter(
                self.transformer["wpe"].weight[:block_size]
            )
        # Also trim the causal masks inside each attention block.
        for block in self.transformer["h"]:
            attn = block.attn
            if hasattr(attn, "bias") and attn.bias is not None:
                attn.bias = attn.bias[:, :, :block_size, :block_size]

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            idx: (B, T) integer token indices.
            targets: (B, T) integer target indices for computing cross-entropy
                loss.  When None, only logits are returned.

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross-entropy loss (+ MoE aux loss), or None if
                ``targets`` is None.
        """
        device = idx.device
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        # Token embeddings.
        tok_emb = self.transformer["wte"](idx)  # (B, T, C)

        # Position embeddings.
        if self.config.pos_embedding == "learned":
            pos = torch.arange(T, dtype=torch.long, device=device)  # (T,)
            pos_emb = self.transformer["wpe"](pos)                   # (T, C)
            x = self.transformer["drop"](tok_emb + pos_emb)
        else:
            # RoPE or no positional embedding: dropout only.
            x = self.transformer["drop"](tok_emb)

        # Transformer blocks with optional gradient checkpointing.
        total_aux_loss = torch.zeros(1, device=device, dtype=x.dtype)
        cfg = self.config

        for i, block in enumerate(self.transformer["h"]):
            use_ckpt = (
                cfg.gradient_checkpointing
                and self.training
                and (i % cfg.checkpoint_every_n_layers == 0)
            )
            if use_ckpt:
                # gradient checkpointing wrapper; use_reentrant=False is the
                # recommended modern API (avoids some quirks with non-Tensor args).
                x, aux = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x, aux = block(x)

            total_aux_loss = total_aux_loss + aux

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            # Flatten for F.cross_entropy.
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            loss = ce_loss + total_aux_loss.squeeze()

        return logits, loss

    # ── Text generation ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Auto-regressively generate tokens.

        Args:
            idx: (B, T) conditioning token indices.
            max_new_tokens: number of new tokens to generate.
            temperature: logit scaling factor.  1.0 = unchanged; < 1.0 sharpens
                the distribution; > 1.0 flattens it.
            top_k: if set, restrict sampling to the top-k most probable tokens.
            top_p: if set, use nucleus (top-p) sampling: keep the smallest set
                of tokens whose cumulative probability exceeds ``top_p``.

        Returns:
            idx: (B, T + max_new_tokens) token indices.
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size if necessary.
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            logits, _ = self(idx_cond)          # (B, T, vocab_size)
            logits = logits[:, -1, :]           # (B, vocab_size)

            if temperature != 1.0:
                logits = logits / temperature

            # ── Top-k filtering ──────────────────────────────────────────────
            if top_k is not None and top_k > 0:
                top_k_clamped = min(top_k, logits.size(-1))
                kth_vals, _ = torch.topk(logits, top_k_clamped)
                threshold = kth_vals[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # ── Top-p (nucleus) filtering ─────────────────────────────────────
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens beyond the nucleus.
                sorted_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(sorted_remove, float("-inf"))
                # Scatter back to original ordering.
                logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    # ── Pretrained weight loading ─────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_type: str,
        override_args: Optional[Dict] = None,
    ) -> "GPT":
        """Load GPT-2 weights from HuggingFace Transformers.

        Args:
            model_type: one of 'gpt2' (124 M), 'gpt2-medium' (355 M),
                'gpt2-large' (774 M), 'gpt2-xl' (1558 M).
            override_args: optional dict of GPTConfig fields to override
                (e.g. ``{'dropout': 0.1}``).

        Returns:
            A GPT instance with HuggingFace weights loaded.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}, (
            f"Unknown model type: {model_type!r}"
        )
        override_args = override_args or {}

        # Only keys that leave the GPT-2 weight layout unchanged may be overridden.
        assert set(override_args) <= {"dropout", "use_lora", "lora_rank", "lora_alpha"}, (
            "Only 'dropout' and LoRA settings may be overridden when loading from pretrained weights."
        )

        # Map model type → GPT-2 config.
        config_args: Dict = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        print(f"Loading weights from pretrained GPT-2: {model_type}")

        config_args["vocab_size"] = 50257   # GPT-2 native vocab size
        config_args["block_size"] = 1024
        config_args["bias"]       = True

        config_args.update(override_args)

        config = GPTConfig(**config_args)
        model = cls(config)

        from transformers import GPT2LMHeadModel  # type: ignore
        model.load_hf_state_dict(GPT2LMHeadModel.from_pretrained(model_type).state_dict())
        return model

    def load_hf_state_dict(self, hf_sd: Dict[str, torch.Tensor]) -> None:
        """Copy a HuggingFace GPT-2 state dict into this model, matching by name.

        HF stores Conv1D weights transposed relative to nn.Linear, and names the
        MLP ``mlp`` where this model uses ``ff``. LoRA parameters are absent from
        the HF dict and keep their initial values.
        """
        sd = self.state_dict()
        transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
        loaded = set()
        for k_hf, v in hf_sd.items():
            if k_hf.endswith((".attn.masked_bias", ".attn.bias", "lm_head.weight")):
                continue  # causal-mask buffers; lm_head is tied to wte
            k = k_hf.replace(".mlp.", ".ff.")
            if k_hf.endswith(transposed):
                v = v.t()
            assert sd[k].shape == v.shape, (
                f"Shape mismatch for {k_hf}: HF {v.shape} vs model {sd[k].shape}"
            )
            with torch.no_grad():
                sd[k].copy_(v)
            loaded.add(k)
        missing = [
            k for k in sd
            if k not in loaded and "lora_" not in k and not k.endswith((".attn.bias", "lm_head.weight"))
        ]
        assert not missing, f"Keys not loaded from HF state dict: {missing}"

    # ── Optimizer factory ────────────────────────────────────────────────────

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """Build an AdamW optimizer with weight-decay parameter groups.

        Following the GPT-3/nanoGPT convention:
        - Weight decay is applied to 2-D (matrix) parameters only.
        - Biases, LayerNorm/RMSNorm gains, and embeddings are excluded.
        - LoRA matrices lora_A and lora_B follow the same rule (2-D → decay).

        Args:
            weight_decay: L2 regularisation coefficient.
            learning_rate: initial learning rate.
            betas: AdamW beta coefficients (beta1, beta2).
            device_type: 'cuda' or 'cpu'; enables fused AdamW on CUDA.

        Returns:
            Configured AdamW optimizer.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params: List[torch.Tensor] = []
        nodecay_params: List[torch.Tensor] = []

        for pn, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        n_decay   = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        print(
            f"Optimizer: {n_decay} params with decay, "
            f"{n_nodecay} params without decay"
        )

        # Use fused AdamW when available (significantly faster on CUDA).
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_kwargs = {"fused": True} if use_fused else {}
        print(f"Using fused AdamW: {use_fused}")

        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_kwargs,
        )

    # ── Profiling helpers ────────────────────────────────────────────────────

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model FLOPs utilisation (MFU).

        Uses the PaLM paper formula for transformer FLOPs, approximating the
        total FLOPs per forward pass as:

            6 * N * T  (for each of the 6 weight matrices per layer)
            + 12 * n_layer * n_head * head_dim * T²  (attention)

        where N is the number of non-embedding parameters, T is the sequence
        length, and head_dim = n_embd / n_head.

        Args:
            fwdbwd_per_iter: number of forward+backward passes per iteration
                (1 for eval, 2–3 for training depending on gradient accumulation).
            dt: wall-clock time per iteration in seconds.

        Returns:
            MFU as a fraction of theoretical peak FLOPs (A100 = 312 TFLOPS bfloat16).
        """
        N = self.get_num_params(non_embedding=True)
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size

        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_iter = flops_per_token * T * fwdbwd_per_iter

        # A100 SXM4 peak: 312e12 FLOP/s in bfloat16.
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12
        return flops_achieved / flops_promised

    # ── LoRA helpers ─────────────────────────────────────────────────────────

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a state dict containing only LoRA parameters.

        Useful for saving lightweight fine-tuning checkpoints.

        Returns:
            Dict mapping parameter names to tensors for all ``lora_A``,
            ``lora_B``, and ``lora_dropout`` parameters.
        """
        lora_keys = {
            k: v
            for k, v in self.state_dict().items()
            if "lora_A" in k or "lora_B" in k
        }
        return lora_keys

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load LoRA weights from a previously saved state dict.

        Only the keys present in *state_dict* are updated; all other parameters
        are left untouched.

        Args:
            state_dict: dict from :meth:`get_lora_state_dict`.
        """
        current_sd = self.state_dict()
        for k, v in state_dict.items():
            if k not in current_sd:
                raise KeyError(f"Unexpected key in LoRA state dict: {k!r}")
            current_sd[k].copy_(v)

    def freeze_base(self) -> None:
        """Freeze every parameter except LoRA matrices. No-op without LoRA."""
        if not self.config.use_lora:
            return
        for name, p in self.named_parameters():
            p.requires_grad_("lora_" in name)
