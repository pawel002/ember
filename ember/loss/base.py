from abc import ABC, abstractmethod

from ember.tensor import Tensor


class Loss(ABC):
    """Base class for losses.

    ``forward`` computes the scalar loss and caches whatever ``backward`` needs;
    ``backward`` returns the gradient of the loss with respect to ``pred``.
    """

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        return self.forward(pred, target)

    @abstractmethod
    def forward(self, pred: Tensor, target: Tensor) -> float:
        raise NotImplementedError

    @abstractmethod
    def backward(self) -> Tensor:
        raise NotImplementedError
