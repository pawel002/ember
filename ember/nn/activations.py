from base import Activation

import ember as em
from ember import Tensor


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        self.y = em.max(x, 0)
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        return grad_y * (self.y > 0)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        self.y = 1.0 / (1.0 + em.exp(-x))
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        return grad_y * (self.y * (1.0 - self.y))
