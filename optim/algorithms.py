"""
Advanced optimization algorithms for nanoGPT Extended.

Provides:
- ``configure_adamw``: AdamW with weight-decay / no-decay parameter groups,
  matching the original nanoGPT pattern and supporting fused CUDA kernels.
- ``LARS``: Layer-wise Adaptive Rate Scaling optimizer (You et al., 2017),
  effective for large-batch training.
- ``get_grad_norm``: Compute the global gradient L2 norm for logging.

References:
- AdamW: Loshchilov & Hutter (2019) https://arxiv.org/abs/1711.05101
- LARS: You et al. (2017) https://arxiv.org/abs/1708.03888
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# AdamW with weight-decay param groups
# ---------------------------------------------------------------------------

def configure_adamw(
    model: nn.Module,
    weight_decay: float,
    lr: float,
    betas: Tuple[float, float],
    device_type: str,
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Configure an AdamW optimiser with separate weight-decay and no-decay groups.

    Following the original nanoGPT convention:
    - **Decay group**: all 2-D+ parameters (weight matrices, embeddings).
      Regularisation is only meaningful for these because 1-D tensors (biases,
      LayerNorm gain/bias) do not benefit from it and can even be harmed.
    - **No-decay group**: 1-D parameters (biases, LayerNorm weights) and any
      parameter explicitly excluded (``requires_grad=False``).

    On CUDA, the fused AdamW kernel (available in PyTorch >= 2.0) is used when
    present, which is significantly faster than the for-loop Python
    implementation and avoids separate kernel launches per tensor.

    Args:
        model:        The model whose parameters should be optimised.
        weight_decay: L2 weight decay coefficient for the decay group.
        lr:           Initial learning rate.
        betas:        Adam ``(beta1, beta2)`` momentum coefficients.
        device_type:  ``'cuda'`` or ``'cpu'``.  Fused kernel only on CUDA.
        eps:          Adam numerical stability epsilon.

    Returns:
        A configured :class:`torch.optim.AdamW` instance.

    Example::

        optimizer = configure_adamw(model, weight_decay=0.1, lr=6e-4,
                                    betas=(0.9, 0.95), device_type='cuda')
    """
    # Gather all parameters that require gradients.
    param_dict = {name: p for name, p in model.named_parameters() if p.requires_grad}

    # Split into weight-decayed and non-decayed groups.
    # Any tensor that is 2-D or higher gets weight decay; 1-D tensors
    # (bias, scale, norm weights) do not.
    decay_params: List[torch.nn.Parameter] = [
        p for p in param_dict.values() if p.dim() >= 2
    ]
    nodecay_params: List[torch.nn.Parameter] = [
        p for p in param_dict.values() if p.dim() < 2
    ]

    optim_groups = [
        {'params': decay_params,   'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]

    # Informational printout (useful for debugging).
    num_decay   = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    print(
        f"[configure_adamw] decay params: {len(decay_params):,} tensors / "
        f"{num_decay:,} elements | "
        f"no-decay params: {len(nodecay_params):,} tensors / "
        f"{num_nodecay:,} elements"
    )

    # Use fused AdamW when on CUDA and PyTorch supports it.
    use_fused = (
        device_type == 'cuda'
        and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    )
    if use_fused:
        print("[configure_adamw] Using fused AdamW kernel.")

    extra_kwargs = {'fused': True} if use_fused else {}

    optimizer = torch.optim.AdamW(
        optim_groups, lr=lr, betas=betas, eps=eps, **extra_kwargs
    )
    return optimizer


# ---------------------------------------------------------------------------
# Gradient norm utility
# ---------------------------------------------------------------------------

def get_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """Compute the global gradient norm across all model parameters.

    This is the same quantity clipped by ``torch.nn.utils.clip_grad_norm_``,
    but without actually clipping.  Logging the pre-clip norm is useful for
    diagnosing training instabilities.

    Args:
        model:     The model whose gradients to measure.
        norm_type: Order of the norm (default 2 = L2 norm).

    Returns:
        The global gradient norm as a Python float.  Returns 0.0 if no
        parameter has a gradient attached.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0

    device = parameters[0].grad.device  # type: ignore[union-attr]
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type).to(device)  # type: ignore[union-attr]
            for p in parameters
        ]),
        norm_type,
    )
    return total_norm.item()


# ---------------------------------------------------------------------------
# LARS Optimizer
# ---------------------------------------------------------------------------

class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling (LARS) optimizer.

    LARS scales the learning rate of each layer by the ratio of the layer's
    weight norm to its gradient norm, multiplied by a global trust coefficient
    ``eta``.  This stabilises large-batch training where the naive SGD
    learning rate would need to be prohibitively small for some layers and
    prohibitively large for others.

    The effective per-layer learning rate is:
        local_lr = eta * ||w|| / (||grad|| + weight_decay * ||w||)
        update   = momentum * v + local_lr * (grad + weight_decay * w)

    Parameters that have ``weight_decay=0`` (e.g. biases, BatchNorm scale)
    bypass the LARS scaling and use the global learning rate directly,
    following the LARS paper's recommendation.

    Args:
        params:          Iterable of parameters or param groups.
        lr:              Global base learning rate.
        momentum:        SGD momentum coefficient (default 0.9).
        weight_decay:    L2 regularisation coefficient (default 0.0).
        eta:             LARS trust coefficient (default 0.001).
        eps:             Small constant to avoid division by zero (default 1e-8).
        exclude_bias_bn: If True, do not apply LARS scaling to bias and 1-D
                         parameters (LayerNorm / BatchNorm).  Recommended.

    Reference:
        Yang You et al. (2017) "Large Batch Training of Convolutional Networks"
        https://arxiv.org/abs/1708.03888

    Example::

        optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9,
                         weight_decay=1e-4, eta=1e-3)
        for x, y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eta: float = 0.001,
        eps: float = 1e-8,
        exclude_bias_bn: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eta < 0.0:
            raise ValueError(f"Invalid LARS trust coefficient eta: {eta}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            eps=eps,
            exclude_bias_bn=exclude_bias_bn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single LARS optimization step.

        Args:
            closure: A closure that re-evaluates the model and returns the
                     loss.  Optional.

        Returns:
            Loss value if ``closure`` is provided, else ``None``.
        """
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr          = group['lr']
            momentum    = group['momentum']
            weight_decay = group['weight_decay']
            eta         = group['eta']
            eps         = group['eps']
            exclude_bn  = group['exclude_bias_bn']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LARS does not support sparse gradients.")

                param_state = self.state[p]

                # Determine whether to apply LARS scaling for this parameter.
                # Biases and 1-D parameters (norm layers) are excluded when
                # exclude_bias_bn=True.
                apply_lars = not (exclude_bn and p.dim() <= 1)

                if apply_lars and weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                if apply_lars:
                    param_norm = p.norm(2)
                    grad_norm  = grad.norm(2)
                    # Local learning rate: LARS scaling factor.
                    if param_norm > 0 and grad_norm > 0:
                        adaptive_lr = eta * param_norm / (grad_norm + eps)
                    else:
                        adaptive_lr = 1.0
                    effective_lr = lr * adaptive_lr
                else:
                    # Skip LARS scaling for bias / BN parameters.
                    effective_lr = lr

                # Momentum update (SGD with momentum).
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p)
                else:
                    buf = param_state['momentum_buffer']

                buf.mul_(momentum).add_(grad, alpha=effective_lr)
                p.add_(-buf)

        return loss
