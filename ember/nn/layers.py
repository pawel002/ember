import math

from base import Layer

import ember as em
from ember import Tensor


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.x = Tensor | None
        self.y = Tensor | None

        self.w: Tensor
        self.b: Tensor | None

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

        # TODO: add em.zeros and em.const
        if self.bias:
            self.bias = em.zeros(size=())
