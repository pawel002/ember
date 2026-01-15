from __future__ import annotations

import math
from typing import Any, Literal, Union

import numpy as np
from numpy.typing import NDArray

from ember._core import (
    # types
    TensorBinaryOp,
    TensorScalarOp,
    TensorUnaryOp,
    _add_scalar,
    _add_tensor,
    _cos,
    _cosh,
    _ctg,
    _ctgh,
    _exponent,
    _from_numpy,
    _gt_scalar,
    _gt_tensor,
    _matmul,
    _max_scalar,
    _max_tensor,
    _min_scalar,
    _min_tensor,
    _mul_scalar,
    _mul_tensor,
    _negate,
    _rsub_scalar,
    _rtruediv_scalar,
    _sin,
    _sinh,
    _sub_scalar,
    _sub_tensor,
    _tan,
    _tanh,
    # tensor and its operators
    _Tensor,
    _transpose,
    _truediv_scalar,
    _truediv_tensor,
)

from .tensor_utils import extract_data_info

Types = Literal["int32", "float32"]
BinaryOpType = Union["Tensor", float, int]
_Types_lookup: dict[type, Types] = {int: "int32", float: "float32"}


class Tensor:
    dtype: Types
    shape: tuple[int, ...]
    _core: _Tensor

    def __init__(self, data: Any):
        shape, dtype_cls, flat_data = extract_data_info(data)
        self.shape = shape
        self.dtype = _Types_lookup.get(dtype_cls, "float32")
        self._core = _Tensor(math.prod(shape))
        self._core._copy_from_list(flat_data)

    @classmethod
    def _from_core(cls, core: _Tensor, shape: tuple[int, ...], dtype: Types) -> Tensor:
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

    def __add__(self, other: BinaryOpType) -> Tensor:
        return _binary_op_wrapper(self, other, "+", _add_tensor, _add_scalar)

    def __radd__(self, other: BinaryOpType) -> Tensor:
        return self + other

    def __sub__(self, other: BinaryOpType) -> Tensor:
        return _binary_op_wrapper(self, other, "-", _sub_tensor, _sub_scalar)

    def __rsub__(self, other: BinaryOpType) -> Tensor:
        return _binary_op_wrapper(self, other, "-", _sub_tensor, _rsub_scalar)

    def __mul__(self, other: BinaryOpType) -> Tensor:
        return _binary_op_wrapper(self, other, "*", _mul_tensor, _mul_scalar)

    def __rmul__(self, other: BinaryOpType) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: BinaryOpType) -> Tensor:
        return _binary_op_wrapper(self, other, "/", _truediv_tensor, _truediv_scalar)

    def __rtruediv__(self, other: BinaryOpType) -> Tensor:
        return _binary_op_wrapper(self, other, "/", _truediv_tensor, _rtruediv_scalar)

    def __gt__(self, other: BinaryOpType) -> Tensor:
        return _binary_op_wrapper(self, other, ">", _gt_tensor, _gt_scalar)

    def __matmul__(self, other: BinaryOpType) -> Tensor:
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

    def to_np(self) -> NDArray:
        result = self._core._to_np()
        return result.reshape(self.shape)

    def to_cpu(self) -> list[Any]:
        return self._core._to_list(self.shape)

    def reshape(self, new_shape: tuple[int, ...]) -> Tensor:
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


def _binary_op_wrapper(
    a: Tensor,
    b: BinaryOpType,
    op_symbol: str,
    tensor_op: TensorBinaryOp,
    float_op: TensorScalarOp,
) -> Tensor:
    result_core = None
    if isinstance(b, Tensor):
        if a.shape != b.shape:
            raise ValueError(
                f"Shape mismatch: {a.shape} cannot {op_symbol} with {b.shape}"
            )
        result_core = tensor_op(a._core, b._core)

    elif isinstance(b, (float, int)):
        result_core = float_op(a._core, float(b))

    if result_core is None:
        raise TypeError(
            f"Unsupported operand type(s) for {op_symbol}: Tensor and '{type(b).__name__}'"
        )

    return Tensor._from_core(result_core, a.shape, a.dtype)


def _unary_op_wrapper(a: Tensor, op_symbol: str, tensor_op: TensorUnaryOp) -> Tensor:
    if not isinstance(a, Tensor):
        raise TypeError(
            f"Ember operator ember.{op_symbol} doesn't support '{type(a).__name__}', only Tensors"
        )

    return Tensor._from_core(tensor_op(a._core), a.shape, a.dtype)


# standalone binary methods
def max(a: Tensor, b: BinaryOpType) -> Tensor:
    return _binary_op_wrapper(a, b, "max()", _max_tensor, _max_scalar)


def min(a: Tensor, b: BinaryOpType) -> Tensor:
    return _binary_op_wrapper(a, b, "min()", _min_tensor, _min_scalar)


# standalone unary methods
def exp(a: Tensor) -> Tensor:
    return _unary_op_wrapper(a, "exp()", _exponent)


# trig
def sin(a: Tensor) -> Tensor:
    return _unary_op_wrapper(a, "sin()", _sin)


def cos(a: Tensor) -> Tensor:
    return _unary_op_wrapper(a, "cos()", _cos)


def tan(a: Tensor) -> Tensor:
    return _unary_op_wrapper(a, "tan()", _tan)


def ctg(a: Tensor) -> Tensor:
    return _unary_op_wrapper(a, "ctg()", _ctg)


# hyper trig
def sinh(a: Tensor) -> Tensor:
    return _unary_op_wrapper(a, "sinh()", _sinh)


def cosh(a: Tensor) -> Tensor:
    return _unary_op_wrapper(a, "cosh()", _cosh)


def tanh(a: Tensor) -> Tensor:
    return _unary_op_wrapper(a, "tanh()", _tanh)


def ctgh(a: Tensor) -> Tensor:
    return _unary_op_wrapper(a, "ctgh()", _ctgh)


# transpose
def T(a: Tensor) -> Tensor:
    if not isinstance(a, Tensor):
        raise TypeError(
            f"Ember operator ember.T() doesn't support '{type(a).__name__}', only Tensors"
        )

    if (dim := len(a.shape)) != 2:
        raise ValueError(f"You can only transpose 2D matrices, not {dim}D")

    new_shape = (a.shape[1], a.shape[0])
    result_core = _transpose(a._core, *a.shape)
    return Tensor._from_core(result_core, new_shape, a.dtype)
