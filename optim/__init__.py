"""
Optimization utilities for nanoGPT Extended.

Re-exports the most commonly used symbols from the sub-modules so that
callers can simply write::

    from optim import configure_adamw, get_lr_schedule, LARS, get_grad_norm
"""

from optim.schedules import (
    get_lr_schedule,
    get_lr,
    cosine_schedule,
    linear_schedule,
    constant_schedule,
    polynomial_schedule,
)

from optim.algorithms import (
    configure_adamw,
    LARS,
    get_grad_norm,
)

__all__ = [
    # Schedules
    'get_lr_schedule',
    'get_lr',
    'cosine_schedule',
    'linear_schedule',
    'constant_schedule',
    'polynomial_schedule',
    # Algorithms
    'configure_adamw',
    'LARS',
    'get_grad_norm',
]
