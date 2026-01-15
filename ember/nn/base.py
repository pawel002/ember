from abc import ABC, abstractmethod
from itertools import chain

from ember.tensor import Tensor


class Layer(ABC):
    x: Tensor | None
    y: Tensor | None

    def __call__(self, x: Tensor, training: bool = True) -> Tensor:
        self.x = x
        self._training = training
        self.y = self.forward(x, training)
        return self.y

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def gradients(self) -> list[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: Tensor, training: bool) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_y: Tensor) -> Tensor:
        raise NotImplementedError


class Activation(Layer):
    def reset(self):
        self.x = None
        self.y = None

    def parameters(self) -> list[Tensor]:
        return []

    def gradients(self) -> list[Tensor]:
        return []


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        super().__init__()
        self.layers: tuple[Layer, ...] = layers

    def reset(self):
        self.x = None
        self.y = None
        for layer in self.layers:
            layer.reset()

    def forward(self, x: Tensor, training: bool) -> Tensor:
        for layer in self.layers:
            x = layer(x, training)
        return x

    def backward(self, grad_y: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad_y = layer.backward(grad_y)
        return grad_y

    def parameters(self) -> list[Tensor]:
        return list(chain.from_iterable(layer.parameters() for layer in self.layers))

    def gradients(self) -> list[Tensor]:
        return list(chain.from_iterable(layer.gradients() for layer in self.layers))
