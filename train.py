"""
Training script for nanoGPT Extended.

Supports single-GPU, multi-GPU DDP via torchrun, gradient accumulation,
mixed precision, torch.compile, cosine/linear/constant LR schedules,
WandB + TensorBoard logging, gradient checkpointing, and early stopping.

Usage
-----
Single GPU::

    python train.py config/train_gpt2.py --learning_rate=6e-4

Multi-GPU (4 GPUs on one node)::

    torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py

Resume from checkpoint::

    python train.py config/train_gpt2.py --init_from=resume --out_dir=out
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# ---------------------------------------------------------------------------
# Top-level configuration variables
# (These are read by configurator.py and can be overridden via CLI or config
#  files.  Keep them at module level so exec()-based overrides work.)
# ---------------------------------------------------------------------------

# I/O
out_dir: str = 'out'
eval_interval: int = 2000
log_interval: int = 1
eval_iters: int = 200
eval_only: bool = False
always_save_checkpoint: bool = True
init_from: str = 'scratch'   # 'scratch' | 'resume' | 'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl'

# WandB
wandb_log: bool = False
wandb_project: str = 'nanogpt'
wandb_run_name: str = 'gpt2'

# Data
dataset: str = 'openwebtext'
gradient_accumulation_steps: int = 5 * 8   # simulates larger batch via micro-steps
batch_size: int = 12
block_size: int = 1024

# Model architecture
n_layer: int = 12
n_head: int = 12
n_embd: int = 768
dropout: float = 0.0
bias: bool = True
# SOTA extensions
attention_type: str = 'standard'       # 'standard' | 'gqa'
n_kv_heads: Optional[int] = None       # for GQA; None => n_head (MHA)
pos_embedding: str = 'learned'         # 'learned' | 'rope'
rope_theta: float = 10000.0
norm_type: str = 'layernorm'           # 'layernorm' | 'rmsnorm'
activation: str = 'gelu'              # 'gelu' | 'swiglu'
use_lora: bool = False
lora_rank: int = 8
lora_alpha: float = 16.0
use_moe: bool = False
num_experts: int = 8
num_experts_per_tok: int = 2
moe_aux_coef: float = 0.01
gradient_checkpointing: bool = False

# Optimiser
learning_rate: float = 6e-4
max_iters: int = 600000
weight_decay: float = 1e-1
beta1: float = 0.9
beta2: float = 0.95
grad_clip: float = 1.0
decay_lr: bool = True
warmup_iters: int = 2000
lr_decay_iters: int = 600000
min_lr: float = 6e-5
lr_schedule: str = 'cosine'            # 'cosine' | 'linear' | 'constant'

# System
device: str = 'auto'                   # 'auto' | 'cuda' | 'cpu' | 'mps'
dtype: str = 'bfloat16'               # 'float32' | 'bfloat16' | 'float16'
compile: bool = True

# Early stopping
patience: Optional[int] = None         # None = disabled

# ---------------------------------------------------------------------------
# Apply overrides from config files and CLI args (nanoGPT configurator style)
# ---------------------------------------------------------------------------
import configurator  # noqa: F401, E402  (side-effect: applies sys.argv to globals)

# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------

ddp: bool = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank: int = int(os.environ['RANK'])
    ddp_local_rank: int = int(os.environ['LOCAL_RANK'])
    ddp_world_size: int = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process: bool = ddp_rank == 0
    seed_offset: int = ddp_rank
    # Each process handles its own gradient accumulation steps, so we scale
    # gradient_accumulation_steps by world_size for consistent effective batch.
    assert gradient_accumulation_steps % ddp_world_size == 0, (
        "gradient_accumulation_steps must be divisible by the number of DDP processes"
    )
    gradient_accumulation_steps //= ddp_world_size
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    seed_offset = 0

# ---------------------------------------------------------------------------
# Device / dtype resolution
# ---------------------------------------------------------------------------

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

device_type: str = 'cuda' if device.startswith('cuda') else device

_DTYPE_MAP = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
}
ptdtype: torch.dtype = _DTYPE_MAP[dtype]

# AMP autocast context — disabled on CPU/MPS which don't support AMP well
if device_type == 'cuda':
    autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
else:
    autocast_ctx = contextlib.nullcontext()

# GradScaler only for fp16 (bf16 doesn't need it)
scaler: Optional[torch.cuda.amp.GradScaler] = None
if device_type == 'cuda' and dtype == 'float16':
    scaler = torch.cuda.amp.GradScaler()

# ---------------------------------------------------------------------------
# Seeding and determinism
# ---------------------------------------------------------------------------

torch.manual_seed(1337 + seed_offset)
if device_type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports from local modules
# ---------------------------------------------------------------------------

from model import GPT, GPTConfig              # noqa: E402
from data.loader import get_batch, get_dataset_path  # noqa: E402
from optim.schedules import get_lr_schedule  # noqa: E402

lr_fn = get_lr_schedule(
    schedule_type=lr_schedule,
    learning_rate=learning_rate,
    min_lr=min_lr,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

data_dir: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', dataset
)


def _get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper around data.loader.get_batch."""
    return get_batch(
        split=split,
        data_dir=data_dir,
        block_size=block_size,
        batch_size=batch_size,
        device=device,
        device_type=device_type,
    )


# ---------------------------------------------------------------------------
# Model construction / loading
# ---------------------------------------------------------------------------

def _build_model_args() -> Dict[str, Any]:
    """Collect architecture hyperparameters into a dict."""
    return dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,    # will be set below
        dropout=dropout,
        attention_type=attention_type,
        n_kv_heads=n_kv_heads,
        pos_embedding=pos_embedding,
        rope_theta=rope_theta,
        norm_type=norm_type,
        activation=activation,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        use_moe=use_moe,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_aux_coef=moe_aux_coef,
        gradient_checkpointing=gradient_checkpointing,
    )


iter_num: int = 0
best_val_loss: float = float('inf')
model_args: Dict[str, Any] = _build_model_args()

if init_from == 'scratch':
    if master_process:
        print(f"Initialising a new model from scratch — dataset: {dataset}")
    # Determine vocab_size from dataset meta if available, otherwise GPT-2 default.
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        import pickle
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size: int = meta['vocab_size']
        if master_process:
            print(f"  Vocab size from {meta_path}: {vocab_size}")
    else:
        vocab_size = 50304  # GPT-2 vocab rounded up to nearest multiple of 64
        if master_process:
            print(f"  No meta.pkl found; using default vocab_size={vocab_size}")
    model_args['vocab_size'] = vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if master_process:
        print(f"Resuming from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Restore model args from checkpoint, then apply any overrides that the
    # user may have specified at the command line (e.g. dropout).
    checkpoint_model_args = checkpoint['model_args']
    # Architecture keys must match exactly.
    for k in ('n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size'):
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    _prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(_prefix):
            state_dict[k[len(_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    if master_process:
        print(f"Initialising from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(
        dropout=dropout,
        attention_type=attention_type,
        n_kv_heads=n_kv_heads,
        pos_embedding=pos_embedding,
        norm_type=norm_type,
        activation=activation,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    model = GPT.from_pretrained(init_from, override_args)
    # Synchronise model_args with what was actually loaded.
    for k in ('n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size'):
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(
        f"Unknown init_from={init_from!r}. "
        "Choose from: 'scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'."
    )

model.to(device)

# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------

if compile:
    if master_process:
        print("Compiling model with torch.compile() — this may take ~1 min...")
    model = torch.compile(model)

# ---------------------------------------------------------------------------
# DDP wrap
# ---------------------------------------------------------------------------

raw_model: nn.Module = model  # handle to the unwrapped model for checkpointing
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

raw_model.freeze_base()
optimizer = raw_model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(beta1, beta2),
    device_type=device_type,
)

if init_from == 'resume' and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint  # free memory

# ---------------------------------------------------------------------------
# Logging: WandB and TensorBoard
# ---------------------------------------------------------------------------

if master_process and wandb_log:
    try:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=model_args)
    except ImportError:
        print("wandb not installed; skipping WandB logging.")
        wandb_log = False

_tb_writer = None
if master_process:
    try:
        from torch.utils.tensorboard import SummaryWriter
        _tb_writer = SummaryWriter(log_dir=os.path.join(out_dir, 'tb_logs'))
    except ImportError:
        pass  # TensorBoard optional


def _log_metrics(metrics: Dict[str, float], step: int) -> None:
    """Log a dict of scalar metrics to WandB and/or TensorBoard."""
    if wandb_log:
        import wandb  # already imported above; this is a local re-import guard
        wandb.log(metrics, step=step)
    if _tb_writer is not None:
        for k, v in metrics.items():
            _tb_writer.add_scalar(k, v, step)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss() -> Dict[str, float]:
    """Run eval_iters forward passes on train and val splits, return mean losses.

    Returns:
        Dict with keys 'train' and 'val', each a float mean loss.
    """
    out: Dict[str, float] = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = _get_batch(split)
            with autocast_ctx:
                if use_moe:
                    _, loss = model(X, Y)
                else:
                    _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def _save_checkpoint(tag: str = 'ckpt') -> None:
    """Save model weights, optimiser state, and training metadata.

    Args:
        tag: Filename stem, e.g. ``'ckpt'`` → ``out/ckpt.pt``.
    """
    ckpt = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': {
            # Snapshot key training hypers alongside the weights.
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'block_size': block_size,
            'max_iters': max_iters,
            'dataset': dataset,
            'dtype': dtype,
            'compile': compile,
        },
    }
    path = os.path.join(out_dir, f'{tag}.pt')
    torch.save(ckpt, path)
    print(f"  Checkpoint saved to {path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

if master_process:
    print(
        f"\n=== nanoGPT Extended — Training ===\n"
        f"  model : {sum(p.numel() for p in raw_model.parameters()) / 1e6:.2f}M params\n"
        f"  device: {device}  |  dtype: {dtype}  |  compile: {compile}\n"
        f"  DDP   : {ddp}  (world_size={ddp_world_size})\n"
        f"  data  : {dataset}  (block={block_size}, batch={batch_size})\n"
        f"  LR    : {learning_rate} (schedule={lr_schedule}, warmup={warmup_iters})\n"
        f"  iters : {max_iters}  eval_every={eval_interval}\n"
    )

if eval_only:
    losses = estimate_loss()
    print(f"eval only — train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
    if ddp:
        destroy_process_group()
    import sys; sys.exit(0)

# Fetch the first batch before entering the loop.
X, Y = _get_batch('train')

# Early stopping state.
_patience_counter: int = 0
_early_stop: bool = False

t0: float = time.time()
local_iter_num: int = 0   # iterations in this process lifetime
running_mfu: float = -1.0  # model FLOPs utilisation (estimated below)

while iter_num < max_iters and not _early_stop:
    # -----------------------------------------------------------------------
    # Set learning rate for this iteration.
    # -----------------------------------------------------------------------
    lr = lr_fn(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # -----------------------------------------------------------------------
    # Periodic evaluation, logging, and checkpointing.
    # -----------------------------------------------------------------------
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        val_loss = losses['val']
        train_loss = losses['train']
        print(
            f"step {iter_num:6d}: train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | lr {lr:.2e}"
        )

        _log_metrics({
            'train/loss': train_loss,
            'val/loss': val_loss,
            'train/lr': lr,
            'train/iter': iter_num,
        }, step=iter_num)

        # Save best checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _patience_counter = 0
            if iter_num > 0:
                _save_checkpoint('ckpt')
        else:
            _patience_counter += 1

        if always_save_checkpoint and iter_num > 0:
            _save_checkpoint(f'ckpt_step{iter_num:07d}')

        # Early stopping check.
        if patience is not None and _patience_counter >= patience:
            print(
                f"Early stopping triggered: val loss did not improve for "
                f"{patience} consecutive evaluations."
            )
            _early_stop = True
            break

    # -----------------------------------------------------------------------
    # Gradient accumulation micro-steps.
    # -----------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    moe_aux_loss_accum: float = 0.0

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # Only sync gradients on the last micro-step for efficiency.
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )

        with autocast_ctx:
            logits, loss = model(X, Y)
            # Scale loss for gradient accumulation.
            loss = loss / gradient_accumulation_steps

        # Prefetch next batch while backward pass runs.
        X, Y = _get_batch('train')

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    # -----------------------------------------------------------------------
    # Gradient clipping.
    # -----------------------------------------------------------------------
    if grad_clip > 0.0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        # Compute norm for logging even if not clipping.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

    # -----------------------------------------------------------------------
    # Optimiser step.
    # -----------------------------------------------------------------------
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    # -----------------------------------------------------------------------
    # Console logging.
    # -----------------------------------------------------------------------
    if iter_num % log_interval == 0 and master_process:
        # Unscaled loss: multiply back by accumulation steps.
        lossf = loss.item() * gradient_accumulation_steps
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        # Rough tokens/sec (each step processes batch_size * block_size tokens
        # across gradient_accumulation_steps * ddp_world_size processes).
        tokens_per_sec = (
            gradient_accumulation_steps
            * ddp_world_size
            * batch_size
            * block_size
            / dt
        )
        print(
            f"iter {iter_num:6d} | loss {lossf:.4f} | lr {lr:.2e} | "
            f"gnorm {grad_norm:.3f} | dt {dt*1000:.0f}ms | "
            f"tok/s {tokens_per_sec:.0f}"
        )
        _log_metrics({
            'train/step_loss': lossf,
            'train/grad_norm': float(grad_norm),
            'train/tokens_per_sec': tokens_per_sec,
        }, step=iter_num)

    iter_num += 1
    local_iter_num += 1

# ---------------------------------------------------------------------------
# Final checkpoint
# ---------------------------------------------------------------------------

if master_process:
    print(f"\nTraining finished at iter {iter_num}. Best val loss: {best_val_loss:.4f}")
    _save_checkpoint('ckpt_final')
    if _tb_writer is not None:
        _tb_writer.close()
    if wandb_log:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass

if ddp:
    destroy_process_group()
