from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ._tensor import _Tensor

# _tensor types
Scalar = float
TensorBinaryOp = Callable[["_Tensor", "_Tensor"], "_Tensor"]
TensorUnaryOp = Callable[["_Tensor"], "_Tensor"]
TensorScalarOp = Callable[["_Tensor", Scalar], "_Tensor"]
