import math

from base import Layer

import ember as em
from ember import Tensor


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

        scale = math.sqrt(self.in_features)
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

    # TODO implement transposition and summation

    # def backward(self, grad_y: Tensor) -> Tensor:
    #     self.grad_w = self.x.T @ grad_y
    #     self.grad_b = grad_y.sum(axis=0)
    #     grad_x = grad_y @ self.w.T
    #     return grad_x
