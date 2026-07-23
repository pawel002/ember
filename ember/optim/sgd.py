import ember as em
from ember.tensor import Tensor

from .base import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent with optional (Polyak) momentum.

    ``v = momentum * v + g`` then ``p -= lr * v``. With ``momentum=0`` this is
    plain gradient descent.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
    ):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = [em.random.zeros(p.shape) for p in self.parameters]

    def apply(self, gradients: list[Tensor]) -> None:
        if len(gradients) != len(self.parameters):
            raise ValueError(
                f"Optimizer expected {len(self.parameters)} gradients, "
                f"but got {len(gradients)}"
            )

        for p, v, g in zip(self.parameters, self.velocities, gradients, strict=True):
            v *= self.momentum
            v += g
            p -= self.lr * v
