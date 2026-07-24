import ember as em
from ember._core import _adam_bias_update, _adamw_step, _adamw_step_dev
from ember.tensor import Tensor

from .base import Optimizer


class AdamW(Optimizer):
    """AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Identical to Adam, except the weight decay is applied directly to the
    parameters (``p *= 1 - lr * weight_decay``) rather than being folded into
    the gradient. With ``capturable=True`` the step counter / bias corrections
    live on-device so it stays exact under CUDA-graph capture.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        capturable: bool = False,
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.capturable = capturable

        self.means = [em.random.zeros(p.shape) for p in self.parameters]
        self.variances = [em.random.zeros(p.shape) for p in self.parameters]

        if capturable:
            self._t = em.random.zeros((1,))
            self._bc = em.random.zeros((2,))

    def apply(self, gradients: list[Tensor]) -> None:
        if len(gradients) != len(self.parameters):
            raise ValueError(
                f"Optimizer expected {len(self.parameters)} gradients, "
                f"but got {len(gradients)}"
            )

        mb1 = 1 - self.beta1
        mb2 = 1 - self.beta2

        if self.capturable:
            _adam_bias_update(self._t._core, self._bc._core, self.beta1, self.beta2)
            for p, m, v, g in zip(
                self.parameters, self.means, self.variances, gradients, strict=True
            ):
                _adamw_step_dev(
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
                    self.weight_decay,
                )
            return

        self.t += 1
        bc1 = 1.0 / (1 - self.beta1**self.t)
        bc2 = 1.0 / (1 - self.beta2**self.t)

        # One fused, in-place kernel launch per parameter.
        for p, m, v, g in zip(
            self.parameters, self.means, self.variances, gradients, strict=True
        ):
            _adamw_step(
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
                self.weight_decay,
            )
