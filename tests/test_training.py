"""Integration tests for training and optimization."""

import pytest
import torch
import math
import tempfile
import os

from model import GPT, GPTConfig
from models.extensions import LoRALinear
from optim.schedules import cosine_schedule, linear_schedule, constant_schedule, get_lr_schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny_model(use_moe=False):
    torch.manual_seed(42)
    cfg = GPTConfig(
        n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=100,
        dropout=0.0, use_moe=use_moe,
        num_experts=4 if use_moe else 8,
    )
    return GPT(cfg), cfg


def make_random_batch(cfg, B=2, T=16):
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))
    return x, y


# ---------------------------------------------------------------------------
# 1. Single forward+backward without error
# ---------------------------------------------------------------------------

class TestForwardBackward:
    def test_basic_step(self):
        model, cfg = make_tiny_model()
        x, y = make_random_batch(cfg)
        _, loss = model(x, y)
        loss.backward()
        assert loss.item() > 0.0

    def test_moe_step(self):
        model, cfg = make_tiny_model(use_moe=True)
        x, y = make_random_batch(cfg, T=8)
        _, loss = model(x, y)
        loss.backward()
        assert loss.item() > 0.0

    def test_gradients_exist(self):
        model, cfg = make_tiny_model()
        x, y = make_random_batch(cfg)
        _, loss = model(x, y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_gradients_finite(self):
        model, cfg = make_tiny_model()
        x, y = make_random_batch(cfg)
        _, loss = model(x, y)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "Non-finite gradient detected"


# ---------------------------------------------------------------------------
# 2. Loss decreases (overfitting tiny batch)
# ---------------------------------------------------------------------------

class TestLossDecreases:
    def test_overfit_tiny_batch(self):
        """Model should overfit a single batch in a few steps."""
        torch.manual_seed(0)
        model, cfg = make_tiny_model()
        model.train()
        x = torch.randint(0, cfg.vocab_size, (4, 16))
        y = torch.randint(0, cfg.vocab_size, (4, 16))
        opt = model.configure_optimizers(
            weight_decay=0.0, learning_rate=1e-2, betas=(0.9, 0.99), device_type="cpu"
        )
        initial_loss = None
        for _ in range(30):
            opt.zero_grad()
            _, loss = model(x, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            opt.step()
        final_loss = loss.item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_lora_overfit(self):
        """LoRA model should also be able to overfit."""
        torch.manual_seed(1)
        cfg = GPTConfig(
            n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=100,
            dropout=0.0, use_lora=True, lora_rank=4,
        )
        model = GPT(cfg)
        model.train()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        opt = model.configure_optimizers(
            weight_decay=0.0, learning_rate=5e-3, betas=(0.9, 0.99), device_type="cpu"
        )
        initial_loss = None
        for _ in range(20):
            opt.zero_grad()
            _, loss = model(x, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            opt.step()
        assert loss.item() < initial_loss


# ---------------------------------------------------------------------------
# 3. Gradient clipping
# ---------------------------------------------------------------------------

class TestGradientClipping:
    def test_clip_reduces_grad_norm(self):
        torch.manual_seed(42)
        model, cfg = make_tiny_model()
        x, y = make_random_batch(cfg)
        _, loss = model(x, y)
        loss.backward()
        max_norm = 1.0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # After clipping, total grad norm should be <= max_norm + eps
        clipped_norm = sum(
            p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
        ) ** 0.5
        assert clipped_norm <= max_norm + 1e-4

    def test_clip_does_not_crash(self):
        model, cfg = make_tiny_model()
        x, y = make_random_batch(cfg)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# 4. LR schedule correctness
# ---------------------------------------------------------------------------

class TestLRSchedules:
    LR = 6e-4
    MIN_LR = 6e-5
    WARMUP = 100
    DECAY = 1000

    def test_cosine_warmup_start(self):
        lr = cosine_schedule(0, self.LR, self.MIN_LR, self.WARMUP, self.DECAY)
        assert lr == pytest.approx(0.0, abs=1e-10)

    def test_cosine_after_warmup(self):
        lr = cosine_schedule(self.WARMUP, self.LR, self.MIN_LR, self.WARMUP, self.DECAY)
        assert lr == pytest.approx(self.LR, rel=1e-4)

    def test_cosine_at_end(self):
        lr = cosine_schedule(self.DECAY, self.LR, self.MIN_LR, self.WARMUP, self.DECAY)
        assert lr == pytest.approx(self.MIN_LR, rel=1e-3)

    def test_cosine_beyond_decay(self):
        lr = cosine_schedule(self.DECAY + 1000, self.LR, self.MIN_LR, self.WARMUP, self.DECAY)
        assert lr == pytest.approx(self.MIN_LR, rel=1e-3)

    def test_linear_warmup_start(self):
        lr = linear_schedule(0, self.LR, self.MIN_LR, self.WARMUP, self.DECAY)
        assert lr == pytest.approx(0.0, abs=1e-10)

    def test_linear_after_warmup(self):
        lr = linear_schedule(self.WARMUP, self.LR, self.MIN_LR, self.WARMUP, self.DECAY)
        assert lr == pytest.approx(self.LR, rel=1e-4)

    def test_constant_schedule(self):
        for it in [0, 50, self.WARMUP, 500]:
            lr = constant_schedule(it, self.LR, self.MIN_LR, self.WARMUP)
            assert lr <= self.LR + 1e-10

    def test_constant_at_warmup(self):
        lr = constant_schedule(self.WARMUP, self.LR, self.MIN_LR, self.WARMUP)
        assert lr == pytest.approx(self.LR, rel=1e-4)

    def test_cosine_monotone_after_warmup(self):
        lrs = [
            cosine_schedule(it, self.LR, self.MIN_LR, self.WARMUP, self.DECAY)
            for it in range(self.WARMUP, self.DECAY + 1, 50)
        ]
        for a, b in zip(lrs, lrs[1:]):
            assert a >= b - 1e-10

    def test_get_lr_schedule_factory(self):
        fn = get_lr_schedule("cosine", self.LR, self.MIN_LR, self.WARMUP, self.DECAY)
        assert callable(fn)
        lr = fn(self.WARMUP)
        assert lr == pytest.approx(self.LR, rel=1e-3)

    def test_get_lr_schedule_linear(self):
        fn = get_lr_schedule("linear", self.LR, self.MIN_LR, self.WARMUP, self.DECAY)
        lr = fn(self.WARMUP)
        assert lr == pytest.approx(self.LR, rel=1e-3)

    def test_get_lr_schedule_constant(self):
        fn = get_lr_schedule("constant", self.LR, self.MIN_LR, self.WARMUP, self.DECAY)
        lr = fn(self.WARMUP + 100)
        assert lr == pytest.approx(self.LR, rel=1e-3)


# ---------------------------------------------------------------------------
# 5. Checkpoint save / load
# ---------------------------------------------------------------------------

class TestCheckpointing:
    def test_save_and_load(self):
        torch.manual_seed(0)
        model, cfg = make_tiny_model()
        opt = model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu"
        )
        # Run one step to populate optimizer state
        x, y = make_random_batch(cfg)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ckpt.pt")
            # Save
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "model_args": vars(cfg),
                "iter_num": 1,
                "best_val_loss": loss.item(),
            }, ckpt_path)
            assert os.path.exists(ckpt_path)

            # Load
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model2, cfg2 = make_tiny_model()
            model2.load_state_dict(checkpoint["model"])
            opt2 = model2.configure_optimizers(
                weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu"
            )
            opt2.load_state_dict(checkpoint["optimizer"])
            assert checkpoint["iter_num"] == 1

            # Outputs should match
            model.eval()
            model2.eval()
            with torch.no_grad():
                logits1, _ = model(x)
                logits2, _ = model2(x)
            assert torch.allclose(logits1, logits2)

    def test_lora_checkpoint(self):
        torch.manual_seed(3)
        cfg = GPTConfig(
            n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=100,
            dropout=0.0, use_lora=True, lora_rank=4,
        )
        model = GPT(cfg)
        lora_sd = model.get_lora_state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lora.pt")
            torch.save(lora_sd, path)
            loaded_sd = torch.load(path, map_location="cpu")

        model2 = GPT(cfg)
        model2.load_lora_state_dict(loaded_sd)
        sd1 = model.get_lora_state_dict()
        sd2 = model2.get_lora_state_dict()
        for k in sd1:
            assert torch.allclose(sd1[k], sd2[k])


# ---------------------------------------------------------------------------
# LoRA freezing
# ---------------------------------------------------------------------------

def make_lora_model():
    torch.manual_seed(42)
    cfg = GPTConfig(
        n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=100,
        dropout=0.0, use_lora=True, lora_rank=4, lora_alpha=8.0,
    )
    model = GPT(cfg)
    model.freeze_base()
    return model, cfg


class TestLoRAFreeze:
    def test_lora_trainable_params_match_rank_formula(self):
        model, _ = make_lora_model()
        layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
        expected = sum(m.rank * (m.in_features + m.out_features) for m in layers)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == expected

    def test_lora_base_weights_unchanged_after_steps(self):
        model, cfg = make_lora_model()
        base = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n}
        opt = model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-2, betas=(0.9, 0.95), device_type="cpu",
        )
        x, y = make_random_batch(cfg)
        for _ in range(10):
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        for n, p in model.named_parameters():
            if "lora_" not in n:
                assert torch.equal(p, base[n]), n
        assert any(p.abs().sum() > 0 for n, p in model.named_parameters() if "lora_B" in n)

    def test_lora_merge_matches_unmerged_forward(self):
        model, cfg = make_lora_model()
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, LoRALinear):
                    m.lora_B.normal_(std=0.02)
        x, _ = make_random_batch(cfg)
        model.eval()
        with torch.no_grad():
            before, _ = model(x)
            for m in model.modules():
                if isinstance(m, LoRALinear):
                    m.merge()
            after, _ = model(x)
        assert (before - after).abs().max().item() < 1e-5

    def test_lora_freeze_is_noop_without_lora(self):
        model, _ = make_tiny_model()
        model.freeze_base()
        assert all(p.requires_grad for p in model.parameters())
