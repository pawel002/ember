import math

import ember as em
from ember.tensor import Tensor

from .base import Loss


class MSELoss(Loss):
    """Mean squared error, averaged over all elements.

    ``loss = mean((pred - target) ** 2)``
    """

    def __init__(self) -> None:
        self.diff: Tensor | None = None
        self.n: int = 0

    def forward(self, pred: Tensor, target: Tensor) -> float:
        self.diff = pred - target
        self.n = math.prod(pred.shape)
        return em.sum(self.diff * self.diff) / self.n

    def backward(self) -> Tensor:
        assert self.diff is not None, "forward() must run before backward()"
        return (2.0 / self.n) * self.diff


class CrossEntropyLoss(Loss):
    """Softmax cross-entropy for classification, averaged over the batch.

    Expects ``pred`` to be raw logits of shape ``(batch, classes)`` and
    ``target`` to be one-hot probabilities of the same shape. The forward pass
    uses the log-sum-exp trick for numerical stability.
    """

    def __init__(self) -> None:
        self.probs: Tensor | None = None
        self.target: Tensor | None = None
        self.n: int = 0

    def forward(self, pred: Tensor, target: Tensor) -> float:
        self.n = pred.shape[0]
        self.target = target

        # log_softmax via the log-sum-exp trick, then cache softmax for backward.
        shifted = pred - em.amax(pred, axis=-1, keepdims=True)
        log_probs = shifted - em.log(em.sum(em.exp(shifted), axis=-1, keepdims=True))
        self.probs = em.exp(log_probs)

        return -em.sum(target * log_probs) / self.n

    def backward(self) -> Tensor:
        assert self.probs is not None, "forward() must run before backward()"
        assert self.target is not None, "forward() must run before backward()"
        return (self.probs - self.target) / self.n
