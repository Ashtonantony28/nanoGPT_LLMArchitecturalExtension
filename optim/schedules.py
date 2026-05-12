"""
Learning rate schedules for nanoGPT Extended.

Provides a family of learning rate schedulers used in modern LLM training:
- Cosine annealing with linear warmup (default nanoGPT)
- Linear decay with linear warmup
- Constant with linear warmup
- Polynomial decay with linear warmup

All schedules share the same warmup phase: the learning rate grows linearly
from 0 to ``learning_rate`` over ``warmup_iters`` steps.

References:
- Cosine schedule: Loshchilov & Hutter (2017), SGDR https://arxiv.org/abs/1608.03983
- Warmup: Vaswani et al. (2017), "Attention Is All You Need"
"""

from __future__ import annotations

import math
from typing import Callable


# ---------------------------------------------------------------------------
# Individual schedule functions
# ---------------------------------------------------------------------------

def cosine_schedule(
    it: int,
    lr: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    """Cosine annealing schedule with linear warmup.

    Three phases:
    1. Linear warmup from 0 → ``lr`` over ``warmup_iters`` steps.
    2. Cosine annealing from ``lr`` → ``min_lr`` over ``lr_decay_iters`` steps.
    3. Constant ``min_lr`` for all steps beyond ``lr_decay_iters``.

    This is the default schedule used in the original nanoGPT.

    Args:
        it:             Current training iteration (0-indexed).
        lr:             Peak learning rate.
        min_lr:         Minimum (floor) learning rate.
        warmup_iters:   Number of warmup steps.
        lr_decay_iters: Total number of steps over which cosine decay runs.

    Returns:
        Learning rate for step ``it``.
    """
    # Phase 1: linear warmup.
    if it < warmup_iters:
        return lr * it / warmup_iters

    # Phase 3: past decay horizon — return floor.
    if it > lr_decay_iters:
        return min_lr

    # Phase 2: cosine annealing between warmup and decay end.
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 → 0
    return min_lr + coeff * (lr - min_lr)


def linear_schedule(
    it: int,
    lr: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    """Linear decay schedule with linear warmup.

    Three phases:
    1. Linear warmup from 0 → ``lr`` over ``warmup_iters`` steps.
    2. Linear decay from ``lr`` → ``min_lr`` over ``lr_decay_iters`` steps.
    3. Constant ``min_lr`` beyond ``lr_decay_iters``.

    Args:
        it:             Current training iteration (0-indexed).
        lr:             Peak learning rate.
        min_lr:         Minimum learning rate.
        warmup_iters:   Number of warmup steps.
        lr_decay_iters: Total steps for the linear decay phase.

    Returns:
        Learning rate for step ``it``.
    """
    if it < warmup_iters:
        return lr * it / warmup_iters

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    return lr - decay_ratio * (lr - min_lr)


def constant_schedule(
    it: int,
    lr: float,
    min_lr: float,
    warmup_iters: int,
) -> float:
    """Constant schedule with linear warmup.

    Two phases:
    1. Linear warmup from 0 → ``lr`` over ``warmup_iters`` steps.
    2. Constant ``lr`` thereafter.

    The ``min_lr`` argument is accepted for API uniformity but is unused.

    Args:
        it:           Current training iteration (0-indexed).
        lr:           Target (constant) learning rate after warmup.
        min_lr:       Unused; present for API uniformity.
        warmup_iters: Number of warmup steps.

    Returns:
        Learning rate for step ``it``.
    """
    if it < warmup_iters:
        return lr * it / warmup_iters
    return lr


def polynomial_schedule(
    it: int,
    lr: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
    power: float = 2.0,
) -> float:
    """Polynomial decay schedule with linear warmup.

    Three phases:
    1. Linear warmup from 0 → ``lr`` over ``warmup_iters`` steps.
    2. Polynomial decay from ``lr`` → ``min_lr`` with exponent ``power``.
    3. Constant ``min_lr`` beyond ``lr_decay_iters``.

    ``power=1`` gives linear decay; ``power=2`` gives quadratic (faster early
    decay); ``power=0.5`` gives square-root decay (slower early decay).

    Args:
        it:             Current training iteration (0-indexed).
        lr:             Peak learning rate.
        min_lr:         Minimum learning rate.
        warmup_iters:   Number of warmup steps.
        lr_decay_iters: Total steps for the decay phase.
        power:          Polynomial exponent (default 2.0).

    Returns:
        Learning rate for step ``it``.
    """
    if it < warmup_iters:
        return lr * it / warmup_iters

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    # decay_ratio goes 0 → 1; (1 - ratio)**power goes 1 → 0.
    coeff = (1.0 - decay_ratio) ** power
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# Backward-compatible dispatcher (matches original nanoGPT's get_lr)
# ---------------------------------------------------------------------------

def get_lr(
    it: int,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    """Cosine LR schedule — drop-in replacement for nanoGPT's ``get_lr``.

    This function provides backward compatibility with the original nanoGPT
    training loop which calls ``get_lr(it)`` with the schedule baked in via
    closure variables.  Here all arguments are explicit.

    Args:
        it:             Current training iteration (0-indexed).
        learning_rate:  Peak learning rate.
        min_lr:         Minimum learning rate.
        warmup_iters:   Number of warmup steps.
        lr_decay_iters: Total steps for cosine annealing.

    Returns:
        Learning rate for step ``it``.
    """
    return cosine_schedule(it, learning_rate, min_lr, warmup_iters, lr_decay_iters)


# ---------------------------------------------------------------------------
# Factory — returns a callable f(it) -> float
# ---------------------------------------------------------------------------

def get_lr_schedule(
    schedule_type: str,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
    power: float = 2.0,
) -> Callable[[int], float]:
    """Return a learning-rate schedule function ``f(it) -> lr``.

    This factory pattern allows training loops to call ``lr = schedule(it)``
    without carrying the schedule hyper-parameters around.

    Args:
        schedule_type:  One of ``'cosine'``, ``'linear'``, ``'constant'``,
                        ``'polynomial'``.
        learning_rate:  Peak learning rate.
        min_lr:         Minimum (floor) learning rate.
        warmup_iters:   Number of linear-warmup steps.
        lr_decay_iters: Total steps over which decay is applied.
        power:          Polynomial exponent (only used for ``'polynomial'``).

    Returns:
        A callable that maps iteration → learning rate.

    Raises:
        ValueError: If ``schedule_type`` is not recognised.

    Example::

        schedule = get_lr_schedule('cosine', 6e-4, 6e-5, 2000, 600000)
        for it in range(max_iters):
            lr = schedule(it)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    """
    schedule_type = schedule_type.lower()

    if schedule_type == 'cosine':
        def _schedule(it: int) -> float:
            return cosine_schedule(it, learning_rate, min_lr, warmup_iters, lr_decay_iters)

    elif schedule_type == 'linear':
        def _schedule(it: int) -> float:
            return linear_schedule(it, learning_rate, min_lr, warmup_iters, lr_decay_iters)

    elif schedule_type == 'constant':
        def _schedule(it: int) -> float:
            return constant_schedule(it, learning_rate, min_lr, warmup_iters)

    elif schedule_type == 'polynomial':
        def _schedule(it: int) -> float:
            return polynomial_schedule(
                it, learning_rate, min_lr, warmup_iters, lr_decay_iters, power
            )

    else:
        raise ValueError(
            f"Unknown schedule_type {schedule_type!r}. "
            "Supported: 'cosine', 'linear', 'constant', 'polynomial'."
        )

    return _schedule
