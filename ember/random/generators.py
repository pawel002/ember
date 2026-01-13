from ember._core import _uniform
from ember.tensor.tensor import Tensor


def uniform(low: float, high: float, size: tuple[int, ...]) -> Tensor:
    return Tensor._from_core(_uniform(low, high, size), size, "float32")


__all__ = ["uniform"]
