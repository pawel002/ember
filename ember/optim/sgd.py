import ember as em
from ember._core import _sgd_step
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

        # One fused, in-place kernel launch per parameter.
        for p, v, g in zip(self.parameters, self.velocities, gradients, strict=True):
            _sgd_step(p._core, g._core, v._core, self.lr, self.momentum)
