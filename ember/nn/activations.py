from ember._core import (
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


class ReLU(Activation):
    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        self.y = Tensor._from_core(_relu(x._core), x.shape, x.dtype)
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self.y is not None, "forward() must run before backward()"
        core = _relu_bwd_tensor(grad_y._core, self.y._core)
        return Tensor._from_core(core, grad_y.shape, grad_y.dtype)


class Sigmoid(Activation):
    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        self.y = Tensor._from_core(_sigmoid(x._core), x.shape, x.dtype)
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self.y is not None, "forward() must run before backward()"
        core = _sigmoid_bwd_tensor(grad_y._core, self.y._core)
        return Tensor._from_core(core, grad_y.shape, grad_y.dtype)


class Tanh(Activation):
    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        self.y = Tensor._from_core(_tanh(x._core), x.shape, x.dtype)
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self.y is not None, "forward() must run before backward()"
        core = _tanh_bwd_tensor(grad_y._core, self.y._core)
        return Tensor._from_core(core, grad_y.shape, grad_y.dtype)


class GELU(Activation):
    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        self.y = Tensor._from_core(_gelu(x._core), x.shape, x.dtype)
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self.x is not None, "forward() must run before backward()"
        core = _gelu_bwd(grad_y._core, self.x._core)
        return Tensor._from_core(core, grad_y.shape, grad_y.dtype)
