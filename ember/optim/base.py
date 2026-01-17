from abc import ABC, abstractmethod

from ember import Tensor


class Optimizer(ABC):
    parameters: list[Tensor]

    @abstractmethod
    def __init__(self, parameters: list[Tensor], *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def apply(self, gradients: list[Tensor]) -> None:
        raise NotImplementedError
