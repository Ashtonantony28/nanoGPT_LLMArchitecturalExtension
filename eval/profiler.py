"""
Performance profiling utilities for nanoGPT Extended.

Provides:
- ``profile_inference``: measure tokens/sec and GPU memory during generation.
- ``profile_training_step``: measure training throughput for a single step.
- ``memory_report``: snapshot of GPU memory allocation.
- ``InferenceProfiler``: context-manager wrapper for block-level profiling.

All functions degrade gracefully on CPU (CUDA-specific stats are omitted).
"""

from __future__ import annotations

import time
import contextlib
from typing import Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Memory snapshot
# ---------------------------------------------------------------------------

def memory_report(device: str = 'cuda') -> Dict[str, float]:
    """Return a dictionary of GPU memory statistics (in megabytes).

    On non-CUDA devices the function returns an empty dict rather than
    raising an exception, so profiling code doesn't need ``if cuda`` guards.

    Args:
        device: The CUDA device string, e.g. ``'cuda'``, ``'cuda:0'``.

    Returns:
        A dict with keys:
        - ``'allocated_mb'``: currently allocated memory.
        - ``'reserved_mb'``: total reserved (cached) memory.
        - ``'max_allocated_mb'``: peak allocated since last reset.
        - ``'free_mb'``: estimated free memory.
        - ``'total_mb'``: total device memory.

    Example::

        report = memory_report('cuda')
        print(f"Allocated: {report['allocated_mb']:.1f} MB")
    """
    if not torch.cuda.is_available() or not device.startswith('cuda'):
        return {}

    torch.cuda.synchronize(device)
    props       = torch.cuda.get_device_properties(device)
    allocated   = torch.cuda.memory_allocated(device)
    reserved    = torch.cuda.memory_reserved(device)
    max_alloc   = torch.cuda.max_memory_allocated(device)
    total       = props.total_memory

    return {
        'allocated_mb':     allocated    / 1e6,
        'reserved_mb':      reserved     / 1e6,
        'max_allocated_mb': max_alloc    / 1e6,
        'free_mb':          (total - reserved) / 1e6,
        'total_mb':         total        / 1e6,
    }


# ---------------------------------------------------------------------------
# Inference profiler
# ---------------------------------------------------------------------------

def profile_inference(
    model: nn.Module,
    config,
    num_tokens: int = 100,
    batch_size: int = 1,
    device: str = 'cuda',
    num_warmup: int = 3,
    num_repeats: int = 10,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Measure inference throughput and memory usage.

    Generates ``num_tokens`` new tokens from a random prompt and reports
    tokens per second averaged over ``num_repeats`` trials after
    ``num_warmup`` warm-up runs (which prime the CUDA kernels and caches).

    Args:
        model:       The GPT model in eval mode.
        config:      Model config object with ``block_size`` and ``vocab_size``.
        num_tokens:  Number of new tokens to generate per trial.
        batch_size:  Number of sequences to generate in parallel.
        device:      Target device string.
        num_warmup:  Warm-up runs excluded from timing.
        num_repeats: Timed trials to average over.
        dtype:       Torch dtype for autocast (ignored on CPU).

    Returns:
        Dict with keys:
        - ``'tokens_per_sec'``: throughput in tokens/second.
        - ``'ms_per_token'``:  latency in milliseconds per token.
        - ``'total_tokens'``:  total tokens generated per run (batch × num_tokens).
        - ``'allocated_mb'``:  GPU memory allocated (0.0 on CPU).
        - ``'reserved_mb'``:  GPU memory reserved (0.0 on CPU).
    """
    model.eval()
    is_cuda = device.startswith('cuda') and torch.cuda.is_available()
    device_type = 'cuda' if is_cuda else 'cpu'

    # Build a fixed random prompt.
    prompt_len = min(32, config.block_size - num_tokens)
    prompt_len = max(1, prompt_len)
    x = torch.randint(0, config.vocab_size, (batch_size, prompt_len), device=device)

    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=dtype)
        if is_cuda
        else contextlib.nullcontext()
    )

    def _run_once() -> float:
        """Generate tokens and return elapsed wall-clock seconds."""
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad(), ctx:
            # Use the model's own generate method when available.
            if hasattr(model, 'generate'):
                model.generate(x, num_tokens, temperature=1.0, top_k=None)
            else:
                # Fallback: manual autoregressive loop.
                inp = x.clone()
                for _ in range(num_tokens):
                    logits, _ = model(inp[:, -config.block_size:])
                    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    inp = torch.cat([inp, next_tok], dim=1)
        if is_cuda:
            torch.cuda.synchronize(device)
        return time.perf_counter() - t0

    # Warm up.
    for _ in range(num_warmup):
        _run_once()

    # Reset peak memory stats before the timed runs.
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    elapsed_times = [_run_once() for _ in range(num_repeats)]
    avg_elapsed   = sum(elapsed_times) / len(elapsed_times)

    total_tokens  = batch_size * num_tokens
    tokens_per_sec = total_tokens / avg_elapsed if avg_elapsed > 0 else float('inf')
    ms_per_token   = (avg_elapsed * 1000) / total_tokens if total_tokens > 0 else float('inf')

    mem = memory_report(device) if is_cuda else {}

    return {
        'tokens_per_sec':    tokens_per_sec,
        'ms_per_token':      ms_per_token,
        'total_tokens':      total_tokens,
        'allocated_mb':      mem.get('allocated_mb', 0.0),
        'reserved_mb':       mem.get('reserved_mb', 0.0),
        'max_allocated_mb':  mem.get('max_allocated_mb', 0.0),
    }


# ---------------------------------------------------------------------------
# Training step profiler
# ---------------------------------------------------------------------------

def profile_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config,
    device: str = 'cuda',
    batch_size: int = 4,
    num_warmup: int = 3,
    num_repeats: int = 10,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """Measure training throughput (tokens/sec) for a single forward+backward step.

    Constructs random input batches of size ``(batch_size, block_size)`` and
    runs the full training step (forward + backward + optimizer step).

    Args:
        model:       The GPT model to profile.
        optimizer:   The optimizer to step.
        config:      Model config with ``block_size`` and ``vocab_size``.
        device:      Target device string.
        batch_size:  Batch size for the profiling step.
        num_warmup:  Warm-up iterations excluded from timing.
        num_repeats: Timed iterations to average over.
        dtype:       Dtype for autocast on CUDA.

    Returns:
        Dict with keys:
        - ``'tokens_per_sec'``:   training throughput.
        - ``'ms_per_step'``:      latency per step in milliseconds.
        - ``'tokens_per_step'``:  tokens processed per step.
        - ``'allocated_mb'``:     GPU memory after profiling.
        - ``'max_allocated_mb'``: peak GPU memory.
    """
    model.train()
    is_cuda = device.startswith('cuda') and torch.cuda.is_available()
    device_type = 'cuda' if is_cuda else 'cpu'

    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=dtype)
        if is_cuda
        else contextlib.nullcontext()
    )

    def _make_batch():
        x = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device=device)
        y = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device=device)
        return x, y

    def _step_once() -> float:
        x, y = _make_batch()
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            # Support both (logits, loss) and (logits, loss, aux) return signatures.
            out = model(x, y)
            if isinstance(out, (tuple, list)):
                loss = out[1]  # second element is typically loss
                # Accumulate any auxiliary losses (e.g. MoE load-balancing).
                if len(out) > 2 and out[2] is not None:
                    loss = loss + out[2]
            else:
                loss = out
        loss.backward()
        optimizer.step()
        if is_cuda:
            torch.cuda.synchronize(device)
        return time.perf_counter() - t0

    # Warm up.
    for _ in range(num_warmup):
        _step_once()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    elapsed_times = [_step_once() for _ in range(num_repeats)]
    avg_elapsed   = sum(elapsed_times) / len(elapsed_times)

    tokens_per_step = batch_size * config.block_size
    tokens_per_sec  = tokens_per_step / avg_elapsed if avg_elapsed > 0 else float('inf')
    ms_per_step     = avg_elapsed * 1000

    mem = memory_report(device) if is_cuda else {}

    return {
        'tokens_per_sec':    tokens_per_sec,
        'ms_per_step':       ms_per_step,
        'tokens_per_step':   tokens_per_step,
        'allocated_mb':      mem.get('allocated_mb', 0.0),
        'max_allocated_mb':  mem.get('max_allocated_mb', 0.0),
    }


# ---------------------------------------------------------------------------
# InferenceProfiler context manager
# ---------------------------------------------------------------------------

class InferenceProfiler:
    """Context manager for timing arbitrary code blocks.

    Measures wall-clock time and (optionally) CUDA memory allocation between
    ``__enter__`` and ``__exit__``.

    Args:
        device:  CUDA device string for memory tracking (optional).
        label:   Descriptive name printed with the report.
        verbose: If True, automatically print the report on exit.

    Example::

        profiler = InferenceProfiler(device='cuda', label='forward pass')
        with profiler:
            output = model(x)
        print(profiler.elapsed_ms, "ms")

        # Or use print_report() for a formatted summary.
        profiler.print_report()
    """

    def __init__(
        self,
        device: str = 'cpu',
        label: str = 'block',
        verbose: bool = False,
    ) -> None:
        self.device  = device
        self.label   = label
        self.verbose = verbose
        self._is_cuda = device.startswith('cuda') and torch.cuda.is_available()

        # Populated on exit.
        self.elapsed_ms:        float = 0.0
        self.allocated_mb:      float = 0.0
        self.reserved_mb:       float = 0.0
        self.max_allocated_mb:  float = 0.0
        self._start_time:       float = 0.0

    def __enter__(self) -> 'InferenceProfiler':
        if self._is_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        if self._is_cuda:
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self._start_time
        self.elapsed_ms = elapsed * 1000

        mem = memory_report(self.device) if self._is_cuda else {}
        self.allocated_mb     = mem.get('allocated_mb', 0.0)
        self.reserved_mb      = mem.get('reserved_mb', 0.0)
        self.max_allocated_mb = mem.get('max_allocated_mb', 0.0)

        if self.verbose:
            self.print_report()

    def print_report(self) -> None:
        """Print a formatted summary of the profiling results."""
        print(f"[InferenceProfiler] {self.label}")
        print(f"  Elapsed:          {self.elapsed_ms:.3f} ms")
        if self._is_cuda:
            print(f"  Allocated:        {self.allocated_mb:.1f} MB")
            print(f"  Peak allocated:   {self.max_allocated_mb:.1f} MB")
            print(f"  Reserved:         {self.reserved_mb:.1f} MB")

    def as_dict(self) -> Dict[str, float]:
        """Return profiling results as a plain dictionary."""
        return {
            'elapsed_ms':        self.elapsed_ms,
            'allocated_mb':      self.allocated_mb,
            'reserved_mb':       self.reserved_mb,
            'max_allocated_mb':  self.max_allocated_mb,
        }
