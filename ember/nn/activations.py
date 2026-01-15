import ember as em
from ember.tensor import Tensor

from .base import Activation


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        self.y = em.max(x, 0)
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self.y, "self.y needed for backward()"

        return grad_y * (self.y > 0.0)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        self.y = 1.0 / (1.0 + em.exp(-x))
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self.y, "self.y needed for backward()"

        return grad_y * (self.y * (1.0 - self.y))


class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        self.y = em.tanh(x)
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self.y, "self.y needed for backward()"

        return grad_y * (1 - self.y * self.y)


class GELU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        self.y = 0.5 * x * (1 + em.tanh(0.8 * x))
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self.y, "self.y needed for backward()"
        assert self.x, "self.x needed for backward()"

        a = 0.8
        return grad_y * ((1.0 + em.tanh(a * self.x)) * (0.5 + a * (self.x - self.y)))
