from collections.abc import Callable

from ._tensor import _Tensor

# tensor operator types
TensorBroadcastedOp = Callable[
    [_Tensor, _Tensor, tuple[int, ...], tuple[int, ...], tuple[int, ...]], _Tensor
]
TensorBinaryOp = Callable[[_Tensor, _Tensor], _Tensor]
TensorUnaryOp = Callable[[_Tensor], _Tensor]
TensorScalarOp = Callable[[_Tensor, float], _Tensor]
