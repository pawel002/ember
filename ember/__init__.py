# expose random module
from . import random
from .tensor.tensor import (
    T,
    Tensor,
    cos,
    cosh,
    ctg,
    ctgh,
    exp,
    max,
    min,
    sin,
    sinh,
    tan,
    tanh,
)

__all__ = [
    "Tensor",
    "max",
    "min",
    "exp",
    "sin",
    "cos",
    "tan",
    "ctg",
    "sinh",
    "cosh",
    "tanh",
    "ctgh",
    "random",
    "T",
]
