"""
Performance benchmarking for nanoGPT Extended.

Measures training throughput, inference speed, memory footprint, and the
effect of key settings such as torch.compile and mixed precision.  Results
are printed as formatted tables.

Usage
-----
::

    # Quick benchmark with default GPT-2 small on the best available device
    python bench.py

    # Specific model size, device, and options
    python bench.py --model_size=gpt2-medium --device=cuda --compile

    # Benchmark all model sizes
    python bench.py --all_sizes

    # Compare attention variants
    python bench.py --compare_attention

    # Profile with PyTorch profiler
    python bench.py --profile --compile
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Model size presets
# ---------------------------------------------------------------------------

#: Named GPT-2 model configurations.  Each entry maps to the constructor args
#: accepted by :class:`GPTConfig`.
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    'gpt2': dict(n_layer=12, n_head=12, n_embd=768),           # 124 M
    'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),   # 350 M
    'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),    # 774 M
    'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),       # 1558 M
    'gpt-mini': dict(n_layer=6, n_head=6, n_embd=384),         # ~10 M (fast CI)
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _detect_device(requested: str) -> Tuple[str, str]:
    """Resolve ``'auto'`` device string and return ``(device, device_type)``.

    Args:
        requested: One of ``'auto'``, ``'cuda'``, ``'cuda:N'``, ``'mps'``, ``'cpu'``.

    Returns:
        Tuple of the resolved device string and its type (``'cuda'``, ``'mps'``,
        or ``'cpu'``).
    """
    if requested == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = requested

    device_type = 'cuda' if device.startswith('cuda') else device
    return device, device_type


def _make_model(
    preset: str,
    block_size: int = 1024,
    vocab_size: int = 50304,
    dropout: float = 0.0,
    attention_type: str = 'standard',
    n_kv_heads: Optional[int] = None,
    pos_embedding: str = 'learned',
    norm_type: str = 'layernorm',
    activation: str = 'gelu',
    use_lora: bool = False,
    use_moe: bool = False,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    gradient_checkpointing: bool = False,
    bias: bool = True,
) -> GPT:
    """Construct a :class:`GPT` model from a named preset.

    Args:
        preset: Key into :data:`MODEL_PRESETS`.
        block_size: Sequence length / context window.
        vocab_size: Vocabulary size.
        dropout: Dropout probability (set to 0 for benchmarking).
        attention_type: ``'standard'`` or ``'gqa'``.
        n_kv_heads: KV head count for GQA.  ``None`` → MHA.
        pos_embedding: ``'learned'`` or ``'rope'``.
        norm_type: ``'layernorm'`` or ``'rmsnorm'``.
        activation: ``'gelu'`` or ``'swiglu'``.
        use_lora: Enable LoRA adaptation.
        use_moe: Enable Mixture-of-Experts feed-forward.
        num_experts: Number of MoE experts.
        num_experts_per_tok: Active experts per token.
        gradient_checkpointing: Enable activation checkpointing.
        bias: Use biases in linear layers.

    Returns:
        Constructed :class:`GPT` model (not yet moved to device).
    """
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset {preset!r}. Choose from: {list(MODEL_PRESETS)}")
    arch = MODEL_PRESETS[preset]
    cfg = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        dropout=dropout,
        attention_type=attention_type,
        n_kv_heads=n_kv_heads,
        pos_embedding=pos_embedding,
        norm_type=norm_type,
        activation=activation,
        use_lora=use_lora,
        use_moe=use_moe,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        gradient_checkpointing=gradient_checkpointing,
        bias=bias,
        **arch,
    )
    return GPT(cfg)


def _param_count_m(model: nn.Module) -> float:
    """Return model parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def _reset_peak_memory(device: str) -> None:
    """Reset the CUDA peak memory stats if on a CUDA device."""
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: str) -> float:
    """Return peak GPU memory allocated in MiB (0 on non-CUDA devices)."""
    if device.startswith('cuda'):
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


def _autocast_ctx(device_type: str, dtype: torch.dtype) -> contextlib.AbstractContextManager:
    """Return an AMP autocast context manager appropriate for *device_type*."""
    if device_type == 'cuda':
        return torch.amp.autocast(device_type='cuda', dtype=dtype)
    return contextlib.nullcontext()


def _make_batch(
    batch_size: int,
    block_size: int,
    vocab_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a random token batch ``(x, y)`` on *device*.

    Args:
        batch_size: Number of sequences.
        block_size: Sequence length.
        vocab_size: Upper bound for random token ids.
        device: Target device.

    Returns:
        ``(x, y)`` — both are :class:`torch.Tensor` of shape
        ``(batch_size, block_size)`` with dtype ``torch.long``.
    """
    x = torch.randint(0, vocab_size, (batch_size, block_size), dtype=torch.long, device=device)
    y = torch.randint(0, vocab_size, (batch_size, block_size), dtype=torch.long, device=device)
    return x, y


def _fmt_row(*cols: str, widths: List[int]) -> str:
    """Format a table row with fixed-width columns.

    Args:
        *cols: Column strings.
        widths: Width for each column.

    Returns:
        Formatted row string.
    """
    return '  '.join(f"{c:<{w}}" for c, w in zip(cols, widths))


def _print_table(headers: List[str], rows: List[List[str]], widths: Optional[List[int]] = None) -> None:
    """Print a left-aligned ASCII table.

    Args:
        headers: Column header strings.
        rows: List of rows, each a list of strings matching headers length.
        widths: Optional explicit column widths.  Computed from content if omitted.
    """
    if widths is None:
        widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = '-+-'.join('-' * w for w in widths)
    print(_fmt_row(*headers, widths=widths))
    print(sep)
    for row in rows:
        print(_fmt_row(*row, widths=widths))


# ---------------------------------------------------------------------------
# Core benchmark routines
# ---------------------------------------------------------------------------

def benchmark_training(
    model: GPT,
    device: str,
    device_type: str,
    batch_size: int = 4,
    block_size: int = 1024,
    num_steps: int = 20,
    warmup_steps: int = 5,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """Measure training throughput (forward + backward + step).

    Args:
        model: The GPT model in training mode.
        device: Target device string.
        device_type: ``'cuda'``, ``'mps'``, or ``'cpu'``.
        batch_size: Micro-batch size.
        block_size: Sequence length.
        num_steps: Total steps to run (including warmup).
        warmup_steps: Steps to discard before timing.
        amp_dtype: Dtype for AMP autocast.

    Returns:
        Dict with keys:

        - ``tokens_per_sec``: Training throughput.
        - ``ms_per_step``: Mean wall-clock time per step in ms.
        - ``peak_mem_mb``: Peak GPU memory in MiB (0 on non-CUDA).
    """
    model.train()
    vocab_size = model.config.vocab_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ctx = _autocast_ctx(device_type, amp_dtype)
    _reset_peak_memory(device)

    step_times: List[float] = []

    for step in range(num_steps):
        x, y = _make_batch(batch_size, block_size, vocab_size, device)
        optimizer.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if device_type == 'cuda':
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        if step >= warmup_steps:
            step_times.append(t1 - t0)

    mean_dt = sum(step_times) / len(step_times)
    tokens_per_sec = (batch_size * block_size) / mean_dt

    return {
        'tokens_per_sec': tokens_per_sec,
        'ms_per_step': mean_dt * 1000,
        'peak_mem_mb': _peak_memory_mb(device),
    }


def benchmark_inference(
    model: GPT,
    device: str,
    device_type: str,
    batch_size: int = 1,
    max_new_tokens: int = 128,
    context_len: int = 64,
    num_runs: int = 5,
    warmup_runs: int = 2,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """Measure autoregressive generation throughput.

    Args:
        model: The GPT model in eval mode.
        device: Target device string.
        device_type: ``'cuda'``, ``'mps'``, or ``'cpu'``.
        batch_size: Batch size for generation.
        max_new_tokens: Number of new tokens to generate per run.
        context_len: Length of the prompt context.
        num_runs: Total runs (including warmup).
        warmup_runs: Runs to discard before timing.
        amp_dtype: Dtype for AMP autocast.

    Returns:
        Dict with keys:

        - ``tokens_per_sec``: Generation throughput.
        - ``ms_per_token``: Mean wall-clock time per generated token in ms.
        - ``peak_mem_mb``: Peak GPU memory in MiB (0 on non-CUDA).
    """
    model.eval()
    vocab_size = model.config.vocab_size
    ctx = _autocast_ctx(device_type, amp_dtype)
    _reset_peak_memory(device)

    run_times: List[float] = []

    prompt = torch.randint(0, vocab_size, (batch_size, context_len), device=device)

    with torch.no_grad():
        for run in range(num_runs):
            t0 = time.perf_counter()
            with ctx:
                model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0, top_k=50)
            if device_type == 'cuda':
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            if run >= warmup_runs:
                run_times.append(t1 - t0)

    mean_dt = sum(run_times) / len(run_times)
    total_tokens = batch_size * max_new_tokens
    tokens_per_sec = total_tokens / mean_dt

    return {
        'tokens_per_sec': tokens_per_sec,
        'ms_per_token': (mean_dt / max_new_tokens) * 1000,
        'peak_mem_mb': _peak_memory_mb(device),
    }


def benchmark_memory_vs_batch(
    preset: str,
    device: str,
    device_type: str,
    block_size: int = 1024,
    batch_sizes: Optional[List[int]] = None,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> List[Dict[str, Any]]:
    """Sweep batch sizes and record peak GPU memory for a given model preset.

    Args:
        preset: Model preset name (key of :data:`MODEL_PRESETS`).
        device: Target device string.
        device_type: ``'cuda'``, ``'mps'``, or ``'cpu'``.
        block_size: Sequence length.
        batch_sizes: Batch sizes to try.  Defaults to ``[1, 2, 4, 8, 16, 32]``.
        amp_dtype: Dtype for AMP autocast.

    Returns:
        List of dicts, each containing ``batch_size`` and ``peak_mem_mb``.
        Entries are omitted if OOM occurs.
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]

    results: List[Dict[str, Any]] = []
    for bs in batch_sizes:
        try:
            gc.collect()
            if device_type == 'cuda':
                torch.cuda.empty_cache()
            model = _make_model(preset).to(device)
            model.eval()
            vocab_size = model.config.vocab_size
            _reset_peak_memory(device)
            x, _ = _make_batch(bs, block_size, vocab_size, device)
            ctx = _autocast_ctx(device_type, amp_dtype)
            with torch.no_grad(), ctx:
                model(x)
            if device_type == 'cuda':
                torch.cuda.synchronize(device)
            peak = _peak_memory_mb(device)
            results.append({'batch_size': bs, 'peak_mem_mb': peak})
            del model
        except torch.cuda.OutOfMemoryError:
            results.append({'batch_size': bs, 'peak_mem_mb': float('inf')})
            break  # larger batches won't fit either
    return results


# ---------------------------------------------------------------------------
# High-level benchmark suites
# ---------------------------------------------------------------------------

def run_training_benchmark(
    args: argparse.Namespace,
    device: str,
    device_type: str,
    amp_dtype: torch.dtype,
) -> None:
    """Run and print training throughput benchmark.

    Args:
        args: Parsed CLI arguments.
        device: Resolved device string.
        device_type: Device type string.
        amp_dtype: AMP dtype to use.
    """
    sizes = list(MODEL_PRESETS.keys()) if args.all_sizes else [args.model_size]
    rows: List[List[str]] = []

    for size in sizes:
        try:
            model = _make_model(size).to(device)
            if args.compile:
                print(f"  Compiling {size}...")
                model = torch.compile(model)  # type: ignore[assignment]
            params = _param_count_m(model)
            result = benchmark_training(
                model,
                device=device,
                device_type=device_type,
                batch_size=args.batch_size,
                block_size=args.block_size,
                num_steps=args.num_steps + args.warmup_steps,
                warmup_steps=args.warmup_steps,
                amp_dtype=amp_dtype,
            )
            rows.append([
                size,
                f"{params:.1f}M",
                f"{result['tokens_per_sec']:,.0f}",
                f"{result['ms_per_step']:.1f}",
                f"{result['peak_mem_mb']:.0f}" if device_type == 'cuda' else 'N/A',
            ])
            del model
            gc.collect()
            if device_type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            rows.append([size, 'ERROR', str(exc)[:30], '', ''])

    print("\n--- Training Throughput ---")
    _print_table(
        ['Model', 'Params', 'Tok/s (train)', 'ms/step', 'Peak GPU MB'],
        rows,
    )


def run_inference_benchmark(
    args: argparse.Namespace,
    device: str,
    device_type: str,
    amp_dtype: torch.dtype,
) -> None:
    """Run and print inference speed benchmark.

    Args:
        args: Parsed CLI arguments.
        device: Resolved device string.
        device_type: Device type string.
        amp_dtype: AMP dtype to use.
    """
    sizes = list(MODEL_PRESETS.keys()) if args.all_sizes else [args.model_size]
    rows: List[List[str]] = []

    for size in sizes:
        try:
            model = _make_model(size).to(device)
            model.eval()
            if args.compile:
                model = torch.compile(model)  # type: ignore[assignment]
            params = _param_count_m(model)
            result = benchmark_inference(
                model,
                device=device,
                device_type=device_type,
                batch_size=1,
                max_new_tokens=args.gen_tokens,
                context_len=args.context_len,
                num_runs=args.num_steps + args.warmup_steps,
                warmup_runs=args.warmup_steps,
                amp_dtype=amp_dtype,
            )
            rows.append([
                size,
                f"{params:.1f}M",
                f"{result['tokens_per_sec']:.1f}",
                f"{result['ms_per_token']:.2f}",
                f"{result['peak_mem_mb']:.0f}" if device_type == 'cuda' else 'N/A',
            ])
            del model
            gc.collect()
            if device_type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            rows.append([size, 'ERROR', str(exc)[:30], '', ''])

    print("\n--- Inference Speed ---")
    _print_table(
        ['Model', 'Params', 'Tok/s (gen)', 'ms/token', 'Peak GPU MB'],
        rows,
    )


def run_attention_comparison(
    args: argparse.Namespace,
    device: str,
    device_type: str,
    amp_dtype: torch.dtype,
) -> None:
    """Compare standard MHA vs Grouped-Query Attention throughput.

    Args:
        args: Parsed CLI arguments.
        device: Resolved device string.
        device_type: Device type string.
        amp_dtype: AMP dtype to use.
    """
    preset = args.model_size
    arch = MODEL_PRESETS[preset]
    n_head = arch['n_head']

    variants: List[Tuple[str, Dict[str, Any]]] = [
        ('MHA (standard)', dict(attention_type='standard', n_kv_heads=None)),
        (f'GQA n_kv={n_head//2}', dict(attention_type='gqa', n_kv_heads=n_head // 2)),
        (f'MQA n_kv=1', dict(attention_type='gqa', n_kv_heads=1)),
    ]

    rows: List[List[str]] = []
    for label, attn_kwargs in variants:
        try:
            model = _make_model(preset, **attn_kwargs).to(device)
            if args.compile:
                model = torch.compile(model)  # type: ignore[assignment]
            result = benchmark_training(
                model,
                device=device,
                device_type=device_type,
                batch_size=args.batch_size,
                block_size=args.block_size,
                num_steps=args.num_steps + args.warmup_steps,
                warmup_steps=args.warmup_steps,
                amp_dtype=amp_dtype,
            )
            rows.append([
                label,
                f"{result['tokens_per_sec']:,.0f}",
                f"{result['ms_per_step']:.1f}",
                f"{result['peak_mem_mb']:.0f}" if device_type == 'cuda' else 'N/A',
            ])
            del model
            gc.collect()
            if device_type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            rows.append([label, 'ERROR', str(exc)[:40], ''])

    print(f"\n--- Attention Variant Comparison ({preset}) ---")
    _print_table(['Attention', 'Tok/s (train)', 'ms/step', 'Peak GPU MB'], rows)


def run_precision_comparison(
    args: argparse.Namespace,
    device: str,
    device_type: str,
) -> None:
    """Compare float32 vs bfloat16 vs float16 training throughput.

    Args:
        args: Parsed CLI arguments.
        device: Resolved device string.
        device_type: Device type string.
    """
    preset = args.model_size
    dtypes: List[Tuple[str, torch.dtype]] = [
        ('float32', torch.float32),
        ('bfloat16', torch.bfloat16),
        ('float16', torch.float16),
    ]

    rows: List[List[str]] = []
    for dtype_name, dtype_t in dtypes:
        try:
            model = _make_model(preset).to(device)
            if dtype_name == 'float32' and device_type == 'cuda':
                # Run without AMP for fp32
                result = benchmark_training(
                    model,
                    device=device,
                    device_type='cpu',  # trick: disable AMP
                    batch_size=args.batch_size,
                    block_size=args.block_size,
                    num_steps=args.num_steps + args.warmup_steps,
                    warmup_steps=args.warmup_steps,
                    amp_dtype=dtype_t,
                )
            else:
                result = benchmark_training(
                    model,
                    device=device,
                    device_type=device_type,
                    batch_size=args.batch_size,
                    block_size=args.block_size,
                    num_steps=args.num_steps + args.warmup_steps,
                    warmup_steps=args.warmup_steps,
                    amp_dtype=dtype_t,
                )
            rows.append([
                dtype_name,
                f"{result['tokens_per_sec']:,.0f}",
                f"{result['ms_per_step']:.1f}",
                f"{result['peak_mem_mb']:.0f}" if device_type == 'cuda' else 'N/A',
            ])
            del model
            gc.collect()
            if device_type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            rows.append([dtype_name, 'ERROR', str(exc)[:40], ''])

    print(f"\n--- Mixed Precision Comparison ({preset}) ---")
    _print_table(['dtype', 'Tok/s (train)', 'ms/step', 'Peak GPU MB'], rows)


def run_compile_comparison(
    args: argparse.Namespace,
    device: str,
    device_type: str,
    amp_dtype: torch.dtype,
) -> None:
    """Compare eager vs torch.compile throughput.

    Args:
        args: Parsed CLI arguments.
        device: Resolved device string.
        device_type: Device type string.
        amp_dtype: AMP dtype.
    """
    preset = args.model_size
    rows: List[List[str]] = []

    for label, do_compile in [('eager', False), ('torch.compile', True)]:
        try:
            model = _make_model(preset).to(device)
            if do_compile:
                print(f"  Compiling model for '{label}' comparison — may take ~1 min...")
                model = torch.compile(model)  # type: ignore[assignment]
            result = benchmark_training(
                model,
                device=device,
                device_type=device_type,
                batch_size=args.batch_size,
                block_size=args.block_size,
                num_steps=args.num_steps + args.warmup_steps,
                warmup_steps=args.warmup_steps,
                amp_dtype=amp_dtype,
            )
            rows.append([
                label,
                f"{result['tokens_per_sec']:,.0f}",
                f"{result['ms_per_step']:.1f}",
                f"{result['peak_mem_mb']:.0f}" if device_type == 'cuda' else 'N/A',
            ])
            del model
            gc.collect()
            if device_type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            rows.append([label, 'ERROR', str(exc)[:40], ''])

    print(f"\n--- torch.compile Effect ({preset}) ---")
    _print_table(['Mode', 'Tok/s (train)', 'ms/step', 'Peak GPU MB'], rows)


def run_memory_sweep(
    args: argparse.Namespace,
    device: str,
    device_type: str,
    amp_dtype: torch.dtype,
) -> None:
    """Print peak GPU memory vs batch size for the selected model.

    Args:
        args: Parsed CLI arguments.
        device: Resolved device string.
        device_type: Device type string.
        amp_dtype: AMP dtype.
    """
    if device_type != 'cuda':
        print("\n--- Memory vs Batch Size (CUDA only) ---")
        print("  Skipped: not running on a CUDA device.")
        return

    preset = args.model_size
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = benchmark_memory_vs_batch(
        preset,
        device=device,
        device_type=device_type,
        block_size=args.block_size,
        batch_sizes=batch_sizes,
        amp_dtype=amp_dtype,
    )

    rows = [
        [str(r['batch_size']), f"{r['peak_mem_mb']:.0f}" if r['peak_mem_mb'] != float('inf') else 'OOM']
        for r in results
    ]
    print(f"\n--- Peak GPU Memory vs Batch Size ({preset}, block_size={args.block_size}) ---")
    _print_table(['Batch Size', 'Peak GPU MB'], rows)


def run_profile(
    args: argparse.Namespace,
    device: str,
    device_type: str,
    amp_dtype: torch.dtype,
) -> None:
    """Run a PyTorch profiler pass and print a summary.

    Args:
        args: Parsed CLI arguments.
        device: Resolved device string.
        device_type: Device type string.
        amp_dtype: AMP dtype.
    """
    try:
        from torch.profiler import profile, ProfilerActivity, record_function
    except ImportError:
        print("torch.profiler not available (requires PyTorch >= 1.9).")
        return

    model = _make_model(args.model_size).to(device)
    model.train()
    if args.compile:
        print("  Compiling model for profiling...")
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    vocab_size = MODEL_PRESETS[args.model_size].get('vocab_size', 50304)  # type: ignore[call-overload]
    ctx = _autocast_ctx(device_type, amp_dtype)
    x, y = _make_batch(args.batch_size, args.block_size, 50304, device)

    activities = [ProfilerActivity.CPU]
    if device_type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with record_function("forward"):
                with ctx:
                    _, loss = model(x, y)
            with record_function("backward"):
                loss.backward()
            optimizer.step()

    print(f"\n--- PyTorch Profiler ({args.model_size}, compile={args.compile}) ---")
    print(prof.key_averages().table(sort_by='cuda_time_total' if device_type == 'cuda' else 'cpu_time_total', row_limit=20))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the benchmarking script."""
    parser = argparse.ArgumentParser(
        description='nanoGPT Extended — performance benchmarking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    parser.add_argument('--model_size', default='gpt2',
                        choices=list(MODEL_PRESETS.keys()),
                        help='Model preset to benchmark.')
    parser.add_argument('--all_sizes', action='store_true',
                        help='Benchmark all model sizes (overrides --model_size for size sweeps).')

    # Hardware
    parser.add_argument('--device', default='auto',
                        help="Device: 'auto', 'cuda', 'cuda:N', 'mps', 'cpu'.")
    parser.add_argument('--dtype', default='bfloat16',
                        choices=['float32', 'bfloat16', 'float16'],
                        help='AMP dtype for mixed-precision benchmarks.')

    # Compilation
    parser.add_argument('--compile', action='store_true',
                        help='Benchmark with torch.compile() enabled.')

    # Batch / sequence
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Micro-batch size for training benchmarks.')
    parser.add_argument('--block_size', type=int, default=1024,
                        help='Sequence (context) length.')
    parser.add_argument('--gen_tokens', type=int, default=128,
                        help='Tokens to generate per run in inference benchmark.')
    parser.add_argument('--context_len', type=int, default=64,
                        help='Prompt length for inference benchmark.')

    # Timing
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of timed steps (excluding warmup).')
    parser.add_argument('--warmup_steps', type=int, default=5,
                        help='Warmup steps to discard before timing.')

    # Benchmark suite selection
    parser.add_argument('--profile', action='store_true',
                        help='Run PyTorch profiler instead of throughput benchmarks.')
    parser.add_argument('--compare_attention', action='store_true',
                        help='Run attention variant (MHA vs GQA vs MQA) comparison.')
    parser.add_argument('--compare_precision', action='store_true',
                        help='Run mixed-precision comparison (fp32 / bf16 / fp16).')
    parser.add_argument('--compare_compile', action='store_true',
                        help='Run eager vs torch.compile comparison.')
    parser.add_argument('--memory_sweep', action='store_true',
                        help='Run batch-size memory sweep (CUDA only).')
    parser.add_argument('--all', action='store_true',
                        help='Run all benchmarks.')

    return parser


def main() -> None:
    """Entry point: parse args, run selected benchmarks, print results."""
    parser = _build_parser()
    args = parser.parse_args()

    device, device_type = _detect_device(args.device)
    amp_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

    if device_type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(
        f"nanoGPT Extended Benchmark\n"
        f"  device     : {device}\n"
        f"  dtype      : {args.dtype}\n"
        f"  model_size : {args.model_size}\n"
        f"  compile    : {args.compile}\n"
        f"  batch_size : {args.batch_size}\n"
        f"  block_size : {args.block_size}\n"
    )

    # Determine which suites to run.
    run_all = args.all
    run_train = not (args.profile or args.compare_attention or args.compare_precision
                     or args.compare_compile or args.memory_sweep) or run_all
    run_infer = run_train
    run_attn = args.compare_attention or run_all
    run_prec = args.compare_precision or run_all
    run_comp = args.compare_compile or run_all
    run_mem = args.memory_sweep or run_all
    run_prof = args.profile

    if run_train:
        run_training_benchmark(args, device, device_type, amp_dtype)
    if run_infer:
        run_inference_benchmark(args, device, device_type, amp_dtype)
    if run_attn:
        run_attention_comparison(args, device, device_type, amp_dtype)
    if run_prec:
        run_precision_comparison(args, device, device_type)
    if run_comp:
        run_compile_comparison(args, device, device_type, amp_dtype)
    if run_mem:
        run_memory_sweep(args, device, device_type, amp_dtype)
    if run_prof:
        run_profile(args, device, device_type, amp_dtype)

    print("\nBenchmark complete.")


if __name__ == '__main__':
    main()
