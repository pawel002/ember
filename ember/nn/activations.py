from base import Activation

from ember import Tensor


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        self.y = x.maximum(0)
        return self.y

    # TODO: fix
    def backward(self, grad_y: Tensor) -> Tensor:
        assert self.y is not None
        return Tensor([1])
