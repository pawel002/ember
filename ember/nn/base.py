from abc import ABC, abstractmethod
from collections.abc import Sequence

from ember.tensor import Tensor


class Layer(ABC):
    """Base class for all layers.

    A layer implements ``forward`` (caching whatever its ``backward`` needs)
    and ``backward`` (returning the gradient w.r.t. its input and storing the
    gradients of its parameters). Calling a layer runs ``forward``.

    Parameters are reported in a fixed order by ``parameters()``, with the
    matching gradients from ``gradients()``:

    - a layer that owns tensors directly (``Linear``, ``LayerNorm``, ...)
      overrides both;
    - a layer built from other layers overrides ``sublayers()`` instead, and
      inherits ``parameters()``/``gradients()`` that concatenate its children's
      in that order, so the two lists always line up;
    - a layer without parameters overrides neither.
    """

    def __call__(self, x: Tensor, training: bool = True) -> Tensor:
        return self.forward(x, training)

    def sublayers(self) -> Sequence["Layer"]:
        """The child layers whose parameters this layer reports, in order."""
        return ()

    def parameters(self) -> list[Tensor]:
        return [p for layer in self.sublayers() for p in layer.parameters()]

    def gradients(self) -> list[Tensor | None]:
        return [g for layer in self.sublayers() for g in layer.gradients()]

    @abstractmethod
    def forward(self, x: Tensor, training: bool) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_y: Tensor) -> Tensor:
        raise NotImplementedError


class Activation(Layer):
    """An element-wise activation function (no parameters)."""


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self.layers: tuple[Layer, ...] = layers

    def sublayers(self) -> Sequence[Layer]:
        return self.layers

    def forward(self, x: Tensor, training: bool) -> Tensor:
        for layer in self.layers:
            x = layer(x, training)
        return x

    def backward(self, grad_y: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad_y = layer.backward(grad_y)
        return grad_y
