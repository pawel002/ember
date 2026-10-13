from ember.tensor import Tensor

from .adam import Adam


class AdamW(Adam):
    """AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Identical to :class:`Adam`, except the weight decay is applied directly to
    the parameters (``p *= 1 - lr * weight_decay``) rather than being folded
    into the gradient. Takes the same ``capturable`` / ``foreach`` options.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        capturable: bool = False,
        foreach: bool = False,
    ):
        super().__init__(parameters, lr, betas, eps, capturable, foreach)
        self.weight_decay = weight_decay
