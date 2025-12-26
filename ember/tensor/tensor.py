from __future__ import annotations
from typing import List, Literal, Any, Dict, Tuple, Callable

import math

from ember._core import (
    _Tensor,
    _from_numpy,
    _add,
    _subtract,
    _matmul,
    _multiply_elementwise,
    _negate,
)
from .tensor_utils import extract_data_info

from numpy.typing import NDArray
import numpy as np

Types = Literal["int32", "float32"]
TensorBinaryOp = Callable[[_Tensor, _Tensor], _Tensor]
_Types_lookup: Dict[type, Types] = {int: "int32", float: "float32"}


class Tensor:
    dtype: Types
    shape: Tuple[int, ...]
    _core: _Tensor

    def __init__(self, data: Any):
        shape, dtype_cls, flat_data = extract_data_info(data)
        self.shape = shape
        self.dtype = _Types_lookup.get(dtype_cls, "float32")
        self._core = _Tensor(math.prod(shape))
        self._core._copy_from_list(flat_data)

    @classmethod
    def _from_core(cls, core: _Tensor, shape: Tuple[int, ...], dtype: Types) -> Tensor:
        obj = cls.__new__(cls)

        obj._core = core
        obj.shape = shape
        obj.dtype = dtype

        return obj

    @classmethod
    def from_np(cls, array: NDArray) -> Tensor:
        obj = cls.__new__(cls)

        # for now force casted to f32
        obj._core = _from_numpy(array.astype(np.float32))
        obj.shape = tuple(array.shape)
        obj.dtype = "float32"

        return obj

    def __add__(self, other: Tensor) -> Tensor:
        return binary_op_wrapper(self, other, _add)

    def __sub__(self, other: Tensor) -> Tensor:
        return binary_op_wrapper(self, other, _subtract)

    def __mul__(self, other: Tensor) -> Tensor:
        return binary_op_wrapper(self, other, _multiply_elementwise)

    def __matmul__(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError(
                f"Unsupported operand type(s) for @: Tensor and '{type(other).__name__}'"
            )

        if (dim_self := len(self.shape)) != 2:
            raise ValueError(f"Matrix A has dim {dim_self}, expected is 2")

        if (dim_other := len(other.shape)) != 2:
            raise ValueError(f"Matrix B has dim {dim_other}, expected is 2")

        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Shape mismatch: {self.shape} cannot multiply {other.shape}"
            )

        result_core = _matmul(
            self._core, other._core, self.shape[0], other.shape[1], self.shape[1]
        )
        result_shape = (self.shape[0], other.shape[1])
        result_dtype = self.dtype

        return Tensor._from_core(result_core, result_shape, result_dtype)

    def __neg__(self):
        return Tensor._from_core(_negate(self._core), self.shape, self.dtype)

    def to_np(self):
        result = self._core._to_np()
        return result.reshape(self.shape)

    def to_cpu(self) -> List[Any]:
        return self._core._to_list(self.shape)

    def reshape(self, new_shape: Tuple[int, ...]) -> Tensor:
        total_elements = math.prod(self.shape)

        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError("Only one dimension can be -1 (inferred)")

            known_prod = -1 * math.prod(new_shape)
            if total_elements % known_prod != 0:
                raise ValueError(
                    f"Cannot reshape size {total_elements} into {new_shape}"
                )

            inferred_dim = total_elements // known_prod
            new_shape = tuple([x if x != -1 else inferred_dim for x in new_shape])

        elif math.prod(new_shape) != total_elements:
            raise ValueError(f"Cannot reshape size {total_elements} into {new_shape}")

        self.shape = new_shape
        return self

    def __repr__(self):
        return f"Tensor({self.to_cpu()})"


def binary_op_wrapper(a: Tensor, b: Tensor, op: TensorBinaryOp) -> Tensor:
    print(op)

    if not isinstance(b, Tensor):
        raise TypeError(
            f"Unsupported operand type(s) for *: Tensor and '{type(b).__name__}'"
        )

    if not a.shape == b.shape:
        raise ValueError(f"Shape mismatch {a.shape} != {b.shape}")

    result_core = op(a._core, b._core)
    result_shape = a.shape
    result_dtype = a.dtype

    return Tensor._from_core(result_core, result_shape, result_dtype)
