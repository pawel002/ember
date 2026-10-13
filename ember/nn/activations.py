from ember._core import (
    TensorBinaryOp,
    TensorUnaryOp,
    _gelu,
    _gelu_bwd,
    _relu,
    _relu_bwd_tensor,
    _sigmoid,
    _sigmoid_bwd_tensor,
    _tanh,
    _tanh_bwd_tensor,
)
from ember.tensor import Tensor

from .base import Activation


class _FromOutput(Activation):
    """An activation whose derivative is a function of its output ``y``: one
    fused kernel each way, and backward reads only (grad, y)."""

    _fwd: TensorUnaryOp  # y = f(x)
    _bwd: TensorBinaryOp  # dx = f'(grad, y)

    def __init__(self):
        self.y: Tensor | None = None

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.y = Tensor._from_core(self._fwd(x._core), x.shape, x.dtype)
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        if self.y is None:
            raise RuntimeError("forward() must run before backward()")
        core = self._bwd(grad_y._core, self.y._core)
        return Tensor._from_core(core, grad_y.shape, grad_y.dtype)


class ReLU(_FromOutput):
    _fwd = staticmethod(_relu)
    _bwd = staticmethod(_relu_bwd_tensor)


class Sigmoid(_FromOutput):
    _fwd = staticmethod(_sigmoid)
    _bwd = staticmethod(_sigmoid_bwd_tensor)


class Tanh(_FromOutput):
    _fwd = staticmethod(_tanh)
    _bwd = staticmethod(_tanh_bwd_tensor)


class GELU(Activation):
    """GELU, ``0.5 * x * (1 + tanh(0.8 * x))``. Its derivative needs the input,
    so this one caches ``x`` instead of ``y``."""

    def __init__(self):
        self.x: Tensor | None = None

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        return Tensor._from_core(_gelu(x._core), x.shape, x.dtype)

    def backward(self, grad_y: Tensor) -> Tensor:
        if self.x is None:
            raise RuntimeError("forward() must run before backward()")
        core = _gelu_bwd(grad_y._core, self.x._core)
        return Tensor._from_core(core, grad_y.shape, grad_y.dtype)
