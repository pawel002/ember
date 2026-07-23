import ember as em
from ember.tensor import Tensor

from .base import Optimizer


class AdamW(Optimizer):
    """AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Identical to Adam, except the weight decay is applied directly to the
    parameters (``p *= 1 - lr * weight_decay``) rather than being folded into
    the gradient.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.means = [em.random.zeros(p.shape) for p in self.parameters]
        self.variances = [em.random.zeros(p.shape) for p in self.parameters]

    def apply(self, gradients: list[Tensor]) -> None:
        if len(gradients) != len(self.parameters):
            raise ValueError(
                f"Optimizer expected {len(self.parameters)} gradients, "
                f"but got {len(gradients)}"
            )

        self.t += 1

        for p, m, v, g in zip(
            self.parameters, self.means, self.variances, gradients, strict=True
        ):
            m *= self.beta1
            m += (1 - self.beta1) * g

            v *= self.beta2
            v += (1 - self.beta2) * (g * g)

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            # decoupled weight decay, then the standard Adam step
            p -= self.lr * self.weight_decay * p
            p -= self.lr * m_hat / (em.sqrt(v_hat) + self.eps)
