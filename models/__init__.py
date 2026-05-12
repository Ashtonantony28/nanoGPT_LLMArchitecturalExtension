"""
models package — attention variants and extension modules.

Import GPT and GPTConfig directly from model.py to avoid circular imports:

    from model import GPT, GPTConfig
    from models.attention import CausalSelfAttention, GroupedQueryAttention, RotaryEmbedding
    from models.extensions import RMSNorm, SwiGLUMLP, LoRALinear, MoEBlock
"""

from models.attention import CausalSelfAttention, GroupedQueryAttention, RotaryEmbedding
from models.extensions import RMSNorm, SwiGLUMLP, LoRALinear, MoEBlock

__all__ = [
    "CausalSelfAttention",
    "GroupedQueryAttention",
    "RotaryEmbedding",
    "RMSNorm",
    "SwiGLUMLP",
    "LoRALinear",
    "MoEBlock",
]
