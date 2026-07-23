from ember._core import _constant, _seed, _uniform
from ember.tensor.tensor import Tensor


def seed(value: int) -> None:
    """Seed the random number generator used by ``uniform``."""
    _seed(value)


def uniform(low: float, high: float, size: tuple[int, ...]) -> Tensor:
    return Tensor._from_core(_uniform(low, high, size), size, "float32")


def constant(value: float, size: tuple[int, ...]) -> Tensor:
    return Tensor._from_core(_constant(value, size), size, "float32")


def zeros(size: tuple[int, ...]) -> Tensor:
    return Tensor._from_core(_constant(0.0, size), size, "float32")


def ones(size: tuple[int, ...]) -> Tensor:
    return Tensor._from_core(_constant(1.0, size), size, "float32")


__all__ = ["seed", "uniform", "zeros", "constant", "ones"]
