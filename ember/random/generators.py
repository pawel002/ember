from ember._core import _seed, _uniform
from ember.tensor import creation
from ember.tensor.tensor import Tensor


def seed(value: int) -> None:
    """Seed the random number generator used by ``uniform``."""
    _seed(value)


def uniform(low: float, high: float, size: tuple[int, ...]) -> Tensor:
    return Tensor._from_core(_uniform(low, high, size), size, "float32")


# Constant fills, kept here for compatibility; prefer ember.full/zeros/ones.
def constant(value: float, size: tuple[int, ...]) -> Tensor:
    """Same as ``ember.full(size, value)``."""
    return creation.full(size, value)


def zeros(size: tuple[int, ...]) -> Tensor:
    """Same as ``ember.zeros(size)``."""
    return creation.zeros(size)


def ones(size: tuple[int, ...]) -> Tensor:
    """Same as ``ember.ones(size)``."""
    return creation.ones(size)


__all__ = ["seed", "uniform", "zeros", "constant", "ones"]
