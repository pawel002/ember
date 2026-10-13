"""Tensors filled with a constant value."""

from collections.abc import Sequence

from ember._core import _constant

from .tensor import Tensor


def full(shape: Sequence[int], fill_value: float) -> Tensor:
    """A new tensor of ``shape`` with every element set to ``fill_value``."""
    shape = tuple(shape)
    return Tensor._from_core(_constant(float(fill_value), shape), shape, "float32")


def zeros(shape: Sequence[int]) -> Tensor:
    """A new tensor of ``shape`` filled with zeros."""
    return full(shape, 0.0)


def ones(shape: Sequence[int]) -> Tensor:
    """A new tensor of ``shape`` filled with ones."""
    return full(shape, 1.0)
