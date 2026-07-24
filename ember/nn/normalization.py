import math

import ember as em
from ember._core import _layernorm_bwd, _layernorm_fwd
from ember.tensor import Tensor

from .base import Layer


class LayerNorm(Layer):
    """Layer normalization over the last dimension.

    Normalizes each length-``dim`` row to zero mean / unit variance, then applies
    a learned per-feature scale (``gamma``) and shift (``beta``):

        y = gamma * (x - mean) / sqrt(var + eps) + beta

    Forward and backward are each a single fused CUDA kernel (the mean/variance
    reduction, normalization and affine are fused; the backward fuses the dx
    computation with the dgamma/dbeta accumulation). Accepts any input shape
    whose last axis is ``dim`` -- e.g. (batch, dim) or (batch, seq, dim).
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        self.mean: Tensor | None = None
        self.rstd: Tensor | None = None

        self.gamma = em.random.ones((self.dim,))
        self.beta = em.random.zeros((self.dim,))

        self.grad_gamma: Tensor | None = None
        self.grad_beta: Tensor | None = None

    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]

    def gradients(self) -> list[Tensor | None]:
        return [self.grad_gamma, self.grad_beta]

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        d = x.shape[-1]
        assert d == self.dim, f"LayerNorm expected last dim {self.dim}, got {d}"
        n = math.prod(x.shape[:-1])

        out_c, mean_c, rstd_c = _layernorm_fwd(
            x._core, self.gamma._core, self.beta._core, n, d, self.eps
        )
        self.mean = Tensor._from_core(mean_c, (n,), x.dtype)
        self.rstd = Tensor._from_core(rstd_c, (n,), x.dtype)
        self.y = Tensor._from_core(out_c, x.shape, x.dtype)
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self.x is not None, "forward() must run before backward()"
        assert self.mean is not None and self.rstd is not None

        d = self.dim
        n = math.prod(self.x.shape[:-1])
        dx_c, dgamma_c, dbeta_c = _layernorm_bwd(
            grad_y._core,
            self.x._core,
            self.gamma._core,
            self.mean._core,
            self.rstd._core,
            n,
            d,
        )
        self.grad_gamma = Tensor._from_core(dgamma_c, (d,), grad_y.dtype)
        self.grad_beta = Tensor._from_core(dbeta_c, (d,), grad_y.dtype)
        return Tensor._from_core(dx_c, self.x.shape, grad_y.dtype)
