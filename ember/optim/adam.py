import ember as em
from ember._core import _adam_step
from ember.tensor import Tensor

from .base import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        self.means = [em.random.zeros(p.shape) for p in self.parameters]
        self.variances = [em.random.zeros(p.shape) for p in self.parameters]

    def apply(self, gradients: list[Tensor]):
        if len(gradients) != len(self.parameters):
            raise ValueError(
                f"Optimizer expected {len(self.parameters)} gradients, "
                f"but got {len(gradients)}"
            )

        self.t += 1
        mb1 = 1 - self.beta1
        mb2 = 1 - self.beta2
        bc1 = 1.0 / (1 - self.beta1**self.t)
        bc2 = 1.0 / (1 - self.beta2**self.t)

        # One fused, in-place kernel launch per parameter (replaces ~10 ops).
        for p, m, v, g in zip(
            self.parameters, self.means, self.variances, gradients, strict=True
        ):
            _adam_step(
                p._core,
                g._core,
                m._core,
                v._core,
                self.lr,
                self.beta1,
                mb1,
                self.beta2,
                mb2,
                self.eps,
                bc1,
                bc2,
            )
