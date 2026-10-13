import math

import ember as em
from ember._core import _matmul_bias
from ember.tensor import Tensor

from ._functional import mm
from .base import Layer


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

        self.x: Tensor | None
        self.y: Tensor | None

        self.w: Tensor
        self.b: Tensor

        self.grad_w: Tensor | None
        self.grad_b: Tensor | None

        self.reset()

    def reset(self):
        self.x = None
        self.y = None

        scale = 1.0 / math.sqrt(self.in_features)
        self.w = em.random.uniform(
            -scale, scale, size=(self.in_features, self.out_features)
        )

        self.b = em.random.zeros(size=(self.out_features,))

        self.grad_w = None
        self.grad_b = None

    def parameters(self) -> list[Tensor]:
        return [self.w, self.b]

    def gradients(self) -> list[Tensor | None]:
        return [self.grad_w, self.grad_b]

    def forward(self, x: Tensor, training: bool) -> Tensor:
        # Any number of leading dimensions: the GEMM runs on the flattened
        # (n, in_features) view (a free reshape), e.g. (batch, seq, dim).
        if not x.shape or x.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear expects inputs of shape (..., {self.in_features}), "
                f"got {x.shape}"
            )
        self.x = x
        lead = x.shape[:-1]
        n, in_f, out_f = math.prod(lead), self.in_features, self.out_features
        # Fused GEMM + bias epilogue in one call (no broadcast-add temporary).
        core = _matmul_bias(x._core, self.w._core, self.b._core, n, out_f, in_f)
        self.y = Tensor._from_core(core, (*lead, out_f), x.dtype)
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        if self.x is None:
            raise RuntimeError("forward() must run before backward()")

        lead = self.x.shape[:-1]
        n, in_f, out_f = math.prod(lead), self.in_features, self.out_features
        x2 = self.x.reshape((n, in_f))
        g2 = grad_y.reshape((n, out_f))

        # Both products need a transposed operand; passing them as cuBLAS OP_T
        # flags reads x / w in place instead of materializing two transposed
        # copies (and their temporaries) per Linear per step.
        self.grad_w = mm(x2, g2, in_f, out_f, n, trans_a=True)
        self.grad_b = em.sum(g2, axis=0)
        return mm(g2, self.w, n, in_f, out_f, trans_b=True).reshape((*lead, in_f))


class Dropout(Layer):
    def __init__(self, p: float):
        assert 0 <= p < 1, "p must be between 0 and 1 (exclusive of 1)"

        self.x: Tensor | None
        self.y: Tensor | None
        self.mask: Tensor | None

        self.p: float = p
        self.scale: float = 1.0 / (1.0 - p)

        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        self.mask = None

    def parameters(self) -> list[Tensor]:
        return []

    def gradients(self) -> list[Tensor | None]:
        return []

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x

        if training:
            self.mask = em.random.uniform(low=0, high=1.0, size=x.shape) > self.p
            self.y = x * self.mask * self.scale
            return self.y

        self.y = x
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self.mask is not None, "self.mask needed for backward()"

        return grad_y * self.mask * self.scale
