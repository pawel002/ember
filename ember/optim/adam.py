import ember as em
from ember import Tensor

from .base import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        parameters: list[Tensor],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.lr: float = lr
        self.beta1: float = betas[0]
        self.beta2: float = betas[1]
        self.eps: float = eps
        self.t: int = 0

        self.parameters: list[Tensor] = parameters
        self.means: list[Tensor] = [
            em.random.zeros(param.shape) for param in self.parameters
        ]
        self.variances: list[Tensor] = [
            em.random.zeros(param.shape) for param in self.parameters
        ]

    def apply(self, gradients: list[Tensor]):
        self.t += 1

        for p, m_p, v_p, grad_p in zip(
            self.parameters, self.means, self.variances, gradients, strict=True
        ):
            m_p *= self.beta1
            m_p += (1 - self.beta1) * grad_p

            v_p *= self.beta2
            v_p += (1 - self.beta2) * grad_p**2

            mhat_p = m_p / (1 - self.beta1**self.t)
            vhat_p = v_p / (1 - self.beta2**self.t)

            p -= self.lr * mhat_p / (self.eps + em.sqrt(vhat_p))
