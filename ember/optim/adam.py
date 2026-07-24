import ember as em
from ember._core import _adam_bias_update, _adam_step, _adam_step_dev
from ember.tensor import Tensor

from .base import Optimizer


class Adam(Optimizer):
    """Adam optimizer.

    With ``capturable=True`` the step counter and bias corrections live on the
    device, so a step captured into a CUDA graph advances them on every replay
    and stays numerically exact (at a small extra per-step cost). Leave it off
    for plain eager training.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        capturable: bool = False,
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.capturable = capturable

        self.means = [em.random.zeros(p.shape) for p in self.parameters]
        self.variances = [em.random.zeros(p.shape) for p in self.parameters]

        if capturable:
            # device-resident step counter (t) and bias corrections [bc1, bc2]
            self._t = em.random.zeros((1,))
            self._bc = em.random.zeros((2,))

    def apply(self, gradients: list[Tensor]):
        if len(gradients) != len(self.parameters):
            raise ValueError(
                f"Optimizer expected {len(self.parameters)} gradients, "
                f"but got {len(gradients)}"
            )

        mb1 = 1 - self.beta1
        mb2 = 1 - self.beta2

        if self.capturable:
            # Advance t and recompute bias corrections on-device (once per step),
            # then a fused in-place update per parameter reading them.
            _adam_bias_update(self._t._core, self._bc._core, self.beta1, self.beta2)
            for p, m, v, g in zip(
                self.parameters, self.means, self.variances, gradients, strict=True
            ):
                _adam_step_dev(
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
                    self._bc._core,
                )
            return

        self.t += 1
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
