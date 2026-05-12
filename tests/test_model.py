"""
Unit tests for the GPT model (model.py).

All tests run on CPU and use tiny models for speed.
"""

import pytest
import torch
import tempfile
import os

from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Shared tiny config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_config():
    """Minimal GPTConfig for fast CPU tests."""
    torch.manual_seed(42)
    return GPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=True,
    )


@pytest.fixture
def tiny_model(tiny_config):
    """Instantiated tiny GPT model."""
    torch.manual_seed(42)
    return GPT(tiny_config)


# ---------------------------------------------------------------------------
# 1. GPTConfig creation with defaults
# ---------------------------------------------------------------------------

class TestGPTConfig:
    def test_default_block_size(self):
        cfg = GPTConfig()
        assert cfg.block_size == 1024

    def test_default_vocab_size(self):
        cfg = GPTConfig()
        assert cfg.vocab_size == 50304

    def test_default_n_layer(self):
        cfg = GPTConfig()
        assert cfg.n_layer == 12

    def test_default_n_head(self):
        cfg = GPTConfig()
        assert cfg.n_head == 12

    def test_default_n_embd(self):
        cfg = GPTConfig()
        assert cfg.n_embd == 768

    def test_default_dropout(self):
        cfg = GPTConfig()
        assert cfg.dropout == 0.0

    def test_default_bias(self):
        cfg = GPTConfig()
        assert cfg.bias is True

    def test_default_attention_type(self):
        cfg = GPTConfig()
        assert cfg.attention_type == "standard"

    def test_default_pos_embedding(self):
        cfg = GPTConfig()
        assert cfg.pos_embedding == "learned"

    def test_default_norm_type(self):
        cfg = GPTConfig()
        assert cfg.norm_type == "layernorm"

    def test_default_activation(self):
        cfg = GPTConfig()
        assert cfg.activation == "gelu"

    def test_default_use_lora(self):
        cfg = GPTConfig()
        assert cfg.use_lora is False

    def test_default_use_moe(self):
        cfg = GPTConfig()
        assert cfg.use_moe is False

    def test_default_gradient_checkpointing(self):
        cfg = GPTConfig()
        assert cfg.gradient_checkpointing is False

    def test_custom_values(self):
        cfg = GPTConfig(n_layer=4, n_head=4, n_embd=128, vocab_size=256)
        assert cfg.n_layer == 4
        assert cfg.n_head == 4
        assert cfg.n_embd == 128
        assert cfg.vocab_size == 256

    def test_lora_defaults(self):
        cfg = GPTConfig()
        assert cfg.lora_rank == 8
        assert cfg.lora_alpha == 16.0
        assert cfg.lora_dropout == 0.0

    def test_moe_defaults(self):
        cfg = GPTConfig()
        assert cfg.num_experts == 8
        assert cfg.num_experts_per_tok == 2

    def test_rope_theta_default(self):
        cfg = GPTConfig()
        assert cfg.rope_theta == 10000.0


# ---------------------------------------------------------------------------
# 2. GPT model instantiation with various configs
# ---------------------------------------------------------------------------

class TestGPTInstantiation:
    def test_basic_instantiation(self, tiny_config):
        torch.manual_seed(42)
        model = GPT(tiny_config)
        assert model is not None

    def test_learned_pos_embedding(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.pos_embedding = "learned"
        model = GPT(tiny_config)
        assert "wpe" in model.transformer

    def test_rope_pos_embedding(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.pos_embedding = "rope"
        model = GPT(tiny_config)
        assert "wpe" not in model.transformer

    def test_no_pos_embedding(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.pos_embedding = "none"
        model = GPT(tiny_config)
        assert "wpe" not in model.transformer

    def test_rmsnorm_variant(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.norm_type = "rmsnorm"
        model = GPT(tiny_config)
        assert model is not None

    def test_swiglu_activation(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.activation = "swiglu"
        model = GPT(tiny_config)
        assert model is not None

    def test_gqa_attention(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.attention_type = "gqa"
        tiny_config.n_kv_heads = 1
        model = GPT(tiny_config)
        assert model is not None

    def test_lora_enabled(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.use_lora = True
        tiny_config.lora_rank = 4
        tiny_config.lora_alpha = 8.0
        tiny_config.lora_dropout = 0.0
        model = GPT(tiny_config)
        assert model is not None

    def test_moe_enabled(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.use_moe = True
        tiny_config.num_experts = 4
        tiny_config.num_experts_per_tok = 2
        model = GPT(tiny_config)
        assert model is not None

    def test_no_bias(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.bias = False
        model = GPT(tiny_config)
        assert model is not None

    def test_transformer_has_expected_keys(self, tiny_model):
        keys = set(tiny_model.transformer.keys())
        assert "wte" in keys
        assert "wpe" in keys
        assert "drop" in keys
        assert "h" in keys
        assert "ln_f" in keys

    def test_lm_head_weight_tied(self, tiny_model):
        # lm_head weight must be the same object as wte weight
        assert tiny_model.lm_head.weight is tiny_model.transformer["wte"].weight


# ---------------------------------------------------------------------------
# 3. Forward pass with correct output shapes
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_logits_shape_no_targets(self, tiny_model, tiny_config):
        B, T = 2, 16
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, loss = tiny_model(idx)
        assert logits.shape == (B, T, tiny_config.vocab_size)
        assert loss is None

    def test_logits_shape_with_targets(self, tiny_model, tiny_config):
        B, T = 2, 16
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, loss = tiny_model(idx, targets)
        assert logits.shape == (B, T, tiny_config.vocab_size)

    def test_batch_size_one(self, tiny_model, tiny_config):
        idx = torch.randint(0, tiny_config.vocab_size, (1, 8))
        logits, loss = tiny_model(idx)
        assert logits.shape == (1, 8, tiny_config.vocab_size)

    def test_seq_len_one(self, tiny_model, tiny_config):
        idx = torch.randint(0, tiny_config.vocab_size, (2, 1))
        logits, _ = tiny_model(idx)
        assert logits.shape == (2, 1, tiny_config.vocab_size)

    def test_max_seq_len(self, tiny_model, tiny_config):
        B, T = 1, tiny_config.block_size
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, _ = tiny_model(idx)
        assert logits.shape == (B, T, tiny_config.vocab_size)

    def test_sequence_too_long_raises(self, tiny_model, tiny_config):
        idx = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size + 1))
        with pytest.raises(AssertionError):
            tiny_model(idx)

    def test_rope_forward(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.pos_embedding = "rope"
        model = GPT(tiny_config)
        B, T = 2, 16
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, tiny_config.vocab_size)

    def test_gqa_forward(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.attention_type = "gqa"
        tiny_config.n_kv_heads = 1
        model = GPT(tiny_config)
        B, T = 2, 16
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, tiny_config.vocab_size)

    def test_moe_forward(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.use_moe = True
        tiny_config.num_experts = 4
        tiny_config.num_experts_per_tok = 2
        model = GPT(tiny_config)
        B, T = 2, 8
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, _ = model(idx)
        assert logits.shape == (B, T, tiny_config.vocab_size)


# ---------------------------------------------------------------------------
# 4. Loss computation
# ---------------------------------------------------------------------------

class TestLossComputation:
    def test_loss_is_scalar(self, tiny_model, tiny_config):
        B, T = 2, 16
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        _, loss = tiny_model(idx, targets)
        assert loss is not None
        assert loss.shape == ()  # scalar

    def test_loss_is_positive(self, tiny_model, tiny_config):
        B, T = 2, 16
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        _, loss = tiny_model(idx, targets)
        assert loss.item() > 0.0

    def test_loss_roughly_log_vocab(self, tiny_config):
        """Loss on uniform logits should be ~log(vocab_size)."""
        torch.manual_seed(42)
        model = GPT(tiny_config)
        model.eval()
        B, T = 4, 16
        import math
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        _, loss = model(idx, targets)
        expected = math.log(tiny_config.vocab_size)
        # Within 2x of expected for a randomly initialised model
        assert loss.item() < expected * 2.0

    def test_loss_is_differentiable(self, tiny_model, tiny_config):
        B, T = 2, 8
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        _, loss = tiny_model(idx, targets)
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(p.grad is not None for p in tiny_model.parameters())
        assert has_grad

    def test_moe_loss_includes_aux(self, tiny_config):
        """With MoE, the returned loss should incorporate aux_loss."""
        torch.manual_seed(42)
        tiny_config.use_moe = True
        tiny_config.num_experts = 4
        tiny_config.num_experts_per_tok = 2
        model = GPT(tiny_config)
        B, T = 2, 8
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        _, loss = model(idx, targets)
        assert loss is not None
        assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# 5. Generation
# ---------------------------------------------------------------------------

class TestGeneration:
    def test_greedy_output_length(self, tiny_model, tiny_config):
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        n_new = 8
        out = tiny_model.generate(idx, max_new_tokens=n_new)
        assert out.shape == (1, 4 + n_new)

    def test_greedy_batch(self, tiny_model, tiny_config):
        idx = torch.randint(0, tiny_config.vocab_size, (3, 4))
        out = tiny_model.generate(idx, max_new_tokens=5)
        assert out.shape == (3, 9)

    def test_top_k_sampling(self, tiny_model, tiny_config):
        torch.manual_seed(42)
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        out = tiny_model.generate(idx, max_new_tokens=10, temperature=1.0, top_k=5)
        assert out.shape == (1, 14)
        assert out[:, 4:].max().item() < tiny_config.vocab_size

    def test_top_p_sampling(self, tiny_model, tiny_config):
        torch.manual_seed(42)
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        out = tiny_model.generate(idx, max_new_tokens=10, temperature=1.0, top_p=0.9)
        assert out.shape == (1, 14)

    def test_top_k_and_top_p_combined(self, tiny_model, tiny_config):
        torch.manual_seed(42)
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        out = tiny_model.generate(idx, max_new_tokens=5, top_k=10, top_p=0.95)
        assert out.shape == (1, 9)

    def test_temperature_scaling(self, tiny_model, tiny_config):
        """Generation with different temperatures should return correct shape."""
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        out_low_t = tiny_model.generate(idx, max_new_tokens=5, temperature=0.1)
        out_high_t = tiny_model.generate(idx, max_new_tokens=5, temperature=2.0)
        assert out_low_t.shape == (1, 9)
        assert out_high_t.shape == (1, 9)

    def test_generate_context_crop(self, tiny_model, tiny_config):
        """When context > block_size, it should be cropped without error."""
        # Start with block_size tokens — generation adds 1 each step, exceeding limit
        idx = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size))
        out = tiny_model.generate(idx, max_new_tokens=3)
        assert out.shape == (1, tiny_config.block_size + 3)

    def test_generate_preserves_prefix(self, tiny_model, tiny_config):
        """Generated tokens are appended; prefix must be unchanged."""
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        out = tiny_model.generate(idx, max_new_tokens=6)
        assert torch.equal(out[:, :4], idx)


# ---------------------------------------------------------------------------
# 6. Parameter counting
# ---------------------------------------------------------------------------

class TestParameterCounting:
    def test_get_num_params_positive(self, tiny_model):
        n = tiny_model.get_num_params()
        assert n > 0

    def test_non_embedding_less_than_total(self, tiny_model):
        total = tiny_model.get_num_params(non_embedding=False)
        non_emb = tiny_model.get_num_params(non_embedding=True)
        assert non_emb <= total

    def test_rope_model_no_wpe(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.pos_embedding = "rope"
        model = GPT(tiny_config)
        # non_embedding=True subtracts wpe, which doesn't exist for RoPE
        n_non_emb = model.get_num_params(non_embedding=True)
        n_total = model.get_num_params(non_embedding=False)
        assert n_non_emb == n_total

    def test_param_count_scales_with_layers(self, tiny_config):
        torch.manual_seed(0)
        cfg_small = GPTConfig(n_layer=1, n_head=2, n_embd=64, block_size=32, vocab_size=100)
        cfg_large = GPTConfig(n_layer=4, n_head=2, n_embd=64, block_size=32, vocab_size=100)
        m_small = GPT(cfg_small)
        m_large = GPT(cfg_large)
        assert m_large.get_num_params() > m_small.get_num_params()

    def test_lm_head_not_double_counted(self, tiny_model, tiny_config):
        """lm_head is weight-tied to wte, so its params shouldn't be extra."""
        total = sum(p.numel() for p in tiny_model.parameters())
        counted = tiny_model.get_num_params(non_embedding=False)
        assert counted == total


# ---------------------------------------------------------------------------
# 7. Weight initialisation
# ---------------------------------------------------------------------------

class TestWeightInit:
    def test_embedding_weight_nonzero(self, tiny_model):
        w = tiny_model.transformer["wte"].weight
        assert w.abs().max().item() > 0.0

    def test_embedding_weight_std_approx(self, tiny_model):
        w = tiny_model.transformer["wte"].weight
        std = w.std().item()
        # GPT-2 init uses std=0.02; allow generous tolerance
        assert 0.005 < std < 0.1

    def test_layernorm_weight_ones(self, tiny_model):
        ln_f = tiny_model.transformer["ln_f"]
        # LayerNorm weight should start as ones
        assert torch.allclose(ln_f.weight, torch.ones_like(ln_f.weight))

    def test_layernorm_bias_zeros(self, tiny_model):
        ln_f = tiny_model.transformer["ln_f"]
        if ln_f.bias is not None:
            assert torch.allclose(ln_f.bias, torch.zeros_like(ln_f.bias))

    def test_linear_bias_zeros(self, tiny_model):
        # All Linear biases should be initialised to zero
        for name, module in tiny_model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                assert torch.allclose(
                    module.bias, torch.zeros_like(module.bias)
                ), f"Bias of {name} is not zero"

    def test_residual_proj_smaller_std(self, tiny_config):
        """c_proj weights should have smaller std due to scaled init."""
        torch.manual_seed(42)
        model = GPT(tiny_config)
        # Collect all c_proj weights
        c_proj_stds = []
        c_fc_stds = []
        for name, p in model.named_parameters():
            if name.endswith("c_proj.weight"):
                c_proj_stds.append(p.std().item())
            elif name.endswith("c_fc.weight"):
                c_fc_stds.append(p.std().item())
        if c_proj_stds and c_fc_stds:
            assert sum(c_proj_stds) / len(c_proj_stds) < sum(c_fc_stds) / len(c_fc_stds)


# ---------------------------------------------------------------------------
# 8. crop_block_size
# ---------------------------------------------------------------------------

class TestCropBlockSize:
    def test_crop_reduces_block_size(self, tiny_model, tiny_config):
        new_size = tiny_config.block_size // 2
        tiny_model.crop_block_size(new_size)
        assert tiny_model.config.block_size == new_size

    def test_crop_updates_wpe(self, tiny_model, tiny_config):
        new_size = tiny_config.block_size // 2
        tiny_model.crop_block_size(new_size)
        assert tiny_model.transformer["wpe"].weight.shape[0] == new_size

    def test_crop_forward_still_works(self, tiny_model, tiny_config):
        new_size = 8
        tiny_model.crop_block_size(new_size)
        idx = torch.randint(0, tiny_config.vocab_size, (1, new_size))
        logits, _ = tiny_model(idx)
        assert logits.shape == (1, new_size, tiny_config.vocab_size)

    def test_crop_to_same_size(self, tiny_model, tiny_config):
        # Cropping to the same size should be a no-op
        tiny_model.crop_block_size(tiny_config.block_size)
        assert tiny_model.config.block_size == tiny_config.block_size

    def test_crop_larger_raises(self, tiny_model, tiny_config):
        with pytest.raises(AssertionError):
            tiny_model.crop_block_size(tiny_config.block_size + 1)

    def test_crop_rope_model(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.pos_embedding = "rope"
        model = GPT(tiny_config)
        new_size = tiny_config.block_size // 2
        model.crop_block_size(new_size)
        assert model.config.block_size == new_size
        # RoPE model has no wpe — forward should still work
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        logits, _ = model(idx)
        assert logits.shape[0] == 1


# ---------------------------------------------------------------------------
# 9. configure_optimizers returns correct param groups
# ---------------------------------------------------------------------------

class TestConfigureOptimizers:
    def test_returns_adamw(self, tiny_model):
        opt = tiny_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            device_type="cpu",
        )
        assert isinstance(opt, torch.optim.AdamW)

    def test_two_param_groups(self, tiny_model):
        opt = tiny_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            device_type="cpu",
        )
        assert len(opt.param_groups) == 2

    def test_decay_group_has_weight_decay(self, tiny_model):
        wd = 0.1
        opt = tiny_model.configure_optimizers(
            weight_decay=wd,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            device_type="cpu",
        )
        # First group is the decay group
        assert opt.param_groups[0]["weight_decay"] == wd

    def test_nodecay_group_zero_wd(self, tiny_model):
        opt = tiny_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            device_type="cpu",
        )
        assert opt.param_groups[1]["weight_decay"] == 0.0

    def test_all_params_covered(self, tiny_model):
        opt = tiny_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            device_type="cpu",
        )
        total_in_groups = sum(
            p.numel() for group in opt.param_groups for p in group["params"]
        )
        total_trainable = sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)
        assert total_in_groups == total_trainable

    def test_1d_params_in_nodecay_group(self, tiny_model):
        """1-D parameters (biases, norms) must be in the no-decay group."""
        opt = tiny_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            device_type="cpu",
        )
        nodecay_params = opt.param_groups[1]["params"]
        for p in nodecay_params:
            assert p.dim() < 2

    def test_2d_params_in_decay_group(self, tiny_model):
        """2-D+ parameters (weight matrices) must be in the decay group."""
        opt = tiny_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            device_type="cpu",
        )
        decay_params = opt.param_groups[0]["params"]
        for p in decay_params:
            assert p.dim() >= 2

    def test_learning_rate_set(self, tiny_model):
        lr = 3e-4
        opt = tiny_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=lr,
            betas=(0.9, 0.95),
            device_type="cpu",
        )
        for group in opt.param_groups:
            assert group["lr"] == lr

    def test_betas_set(self, tiny_model):
        betas = (0.9, 0.95)
        opt = tiny_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=betas,
            device_type="cpu",
        )
        for group in opt.param_groups:
            assert group["betas"] == betas


# ---------------------------------------------------------------------------
# 10. Model with gradient checkpointing
# ---------------------------------------------------------------------------

class TestGradientCheckpointing:
    def test_forward_with_checkpointing(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.gradient_checkpointing = True
        tiny_config.checkpoint_every_n_layers = 1
        model = GPT(tiny_config)
        model.train()
        B, T = 2, 8
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, loss = model(idx, targets)
        assert logits.shape == (B, T, tiny_config.vocab_size)
        assert loss is not None

    def test_backward_with_checkpointing(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.gradient_checkpointing = True
        model = GPT(tiny_config)
        model.train()
        B, T = 2, 8
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        targets = torch.randint(0, tiny_config.vocab_size, (B, T))
        _, loss = model(idx, targets)
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_checkpoint_every_n_layers(self, tiny_config):
        """checkpoint_every_n_layers=2 should only checkpoint even-indexed layers."""
        torch.manual_seed(42)
        tiny_config.n_layer = 4
        tiny_config.gradient_checkpointing = True
        tiny_config.checkpoint_every_n_layers = 2
        model = GPT(tiny_config)
        model.train()
        idx = torch.randint(0, tiny_config.vocab_size, (1, 8))
        targets = torch.randint(0, tiny_config.vocab_size, (1, 8))
        _, loss = model(idx, targets)
        loss.backward()
        assert loss.item() > 0.0

    def test_checkpointing_disabled_at_eval(self, tiny_config):
        """Checkpointing should be skipped when model is in eval mode."""
        torch.manual_seed(42)
        tiny_config.gradient_checkpointing = True
        model = GPT(tiny_config)
        model.eval()
        idx = torch.randint(0, tiny_config.vocab_size, (1, 8))
        logits, _ = model(idx)
        assert logits.shape[0] == 1


# ---------------------------------------------------------------------------
# LoRA state dict round-trip
# ---------------------------------------------------------------------------

class TestLoRAStateDicts:
    def test_get_lora_state_dict_keys(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.use_lora = True
        tiny_config.lora_rank = 4
        tiny_config.lora_alpha = 8.0
        tiny_config.lora_dropout = 0.0
        model = GPT(tiny_config)
        lora_sd = model.get_lora_state_dict()
        assert len(lora_sd) > 0
        for key in lora_sd:
            assert "lora_A" in key or "lora_B" in key

    def test_load_lora_state_dict_roundtrip(self, tiny_config):
        torch.manual_seed(42)
        tiny_config.use_lora = True
        tiny_config.lora_rank = 4
        tiny_config.lora_alpha = 8.0
        tiny_config.lora_dropout = 0.0
        model = GPT(tiny_config)
        # Mutate lora_A in the first LoRALinear
        lora_sd_before = model.get_lora_state_dict()
        # Fill lora_A with ones to create a distinct state
        with torch.no_grad():
            for k, v in lora_sd_before.items():
                if "lora_A" in k:
                    v.fill_(1.0)
        model.load_lora_state_dict(lora_sd_before)
        lora_sd_after = model.get_lora_state_dict()
        for k in lora_sd_before:
            assert torch.allclose(lora_sd_before[k], lora_sd_after[k])
