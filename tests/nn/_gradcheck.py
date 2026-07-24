"""Finite-difference gradient checking helpers for the hand-written backward
passes. Ember is float32-only, so tolerances are necessarily loose; inputs are
kept small and smooth to keep the central-difference estimate well-conditioned.
"""

from __future__ import annotations

import numpy as np

from ember import Tensor


def _loss(out: Tensor, c_np: np.ndarray) -> float:
    # A scalar loss linear in the output => dL/dout == c (known exactly), which
    # is what we feed to backward().
    return float((out.to_np() * c_np).sum())


def numeric_grad_input(
    forward, x_np: np.ndarray, c_np: np.ndarray, eps: float = 1e-2
) -> np.ndarray:
    """Central-difference dL/dx for L = sum(forward(x) * c)."""
    grad = np.zeros_like(x_np, dtype=np.float64)
    flat = x_np.reshape(-1)
    for i in range(flat.size):
        xp = x_np.copy()
        xm = x_np.copy()
        xp.reshape(-1)[i] += eps
        xm.reshape(-1)[i] -= eps
        lp = _loss(forward(Tensor.from_np(xp)), c_np)
        lm = _loss(forward(Tensor.from_np(xm)), c_np)
        grad.reshape(-1)[i] = (lp - lm) / (2 * eps)
    return grad


def numeric_grad_param(
    forward_once, param: Tensor, c_np: np.ndarray, eps: float = 1e-2
) -> np.ndarray:
    """Central-difference dL/dparam. ``param`` is mutated in place (and restored)
    via copy_from_numpy; ``forward_once()`` recomputes the output tensor."""
    base = param.to_np().copy()
    grad = np.zeros_like(base, dtype=np.float64)
    flat = base.reshape(-1)
    for i in range(flat.size):
        pp = base.copy()
        pp.reshape(-1)[i] += eps
        param.copy_from_numpy(pp)
        lp = _loss(forward_once(), c_np)

        pm = base.copy()
        pm.reshape(-1)[i] -= eps
        param.copy_from_numpy(pm)
        lm = _loss(forward_once(), c_np)

        grad.reshape(-1)[i] = (lp - lm) / (2 * eps)
    param.copy_from_numpy(base)
    return grad
