"""NumPy reference implementations shared by the tests."""

import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over ``axis``."""
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def adam_step(p, g, m, v, t, lr, b1, b2, eps, weight_decay=0.0):
    """One Adam step (AdamW with ``weight_decay`` > 0): the new (p, m, v)."""
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g**2
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    p = p - lr * weight_decay * p
    return p - lr * m_hat / (np.sqrt(v_hat) + eps), m, v
