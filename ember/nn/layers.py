import math

import ember as em
from ember.tensor import Tensor

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
        self.x = x
        self.y = self.b + x @ self.w
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self.x, "self.x needed for backward()"
        assert self.w, "self.w needed for backward()"

        self.grad_w = em.T(self.x) @ grad_y
        self.grad_b = em.sum(grad_y, axis=0)  # type: ignore
        grad_x = grad_y @ em.T(self.w)
        return grad_x
