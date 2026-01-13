from collections.abc import Callable

from ._tensor import _Tensor

# tensor operator types
Scalar = float
TensorBinaryOp = Callable[[_Tensor, _Tensor], _Tensor]
TensorUnaryOp = Callable[[_Tensor], _Tensor]
TensorScalarOp = Callable[[_Tensor, Scalar], _Tensor]
