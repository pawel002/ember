import math

import numpy as np

import ember as em
from ember._core import _embedding_bwd, _embedding_fwd
from ember.tensor import Tensor

from .base import Layer


class Embedding(Layer):
    """A lookup table mapping integer ids to dense vectors.

    ``forward`` takes an integer index array (a NumPy array or nested list of
    ints, any shape) and returns a tensor of shape ``(*idx.shape, embedding_dim)``
    gathered from the weight rows. The gather (forward) and the scatter-add of
    gradients into the used rows (backward) are each a single CUDA kernel.

    Unlike the other layers the input is integer ids rather than a float Tensor,
    so there is no gradient w.r.t. the input; ``backward`` returns ``None`` and
    only populates ``grad_weight``.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reset()

    def reset(self):
        self.idx: np.ndarray | None = None
        self.y = None

        # Small uniform init (the backend has no normal sampler yet).
        scale = 1.0 / math.sqrt(self.embedding_dim)
        self.weight = em.random.uniform(
            -scale, scale, size=(self.num_embeddings, self.embedding_dim)
        )
        self.grad_weight: Tensor | None = None

    def parameters(self) -> list[Tensor]:
        return [self.weight]

    def gradients(self) -> list[Tensor | None]:
        return [self.grad_weight]

    def forward(self, idx, training: bool = True) -> Tensor:
        idx = np.ascontiguousarray(idx, dtype=np.int32)
        self.idx = idx
        out_c = _embedding_fwd(self.weight._core, idx.ravel(), self.embedding_dim)
        out_shape = (*idx.shape, self.embedding_dim)
        self.y = Tensor._from_core(out_c, out_shape, "float32")
        return self.y

    # Input is integer ids, which have no gradient, so this returns None rather
    # than a dx tensor (unlike other layers). Embedding is a leaf/input layer.
    def backward(self, grad_y: Tensor) -> None:  # type: ignore[override]
        assert self.idx is not None, "forward() must run before backward()"
        dweight_c = _embedding_bwd(
            grad_y._core,
            self.idx.ravel(),
            self.num_embeddings,
            self.embedding_dim,
        )
        self.grad_weight = Tensor._from_core(
            dweight_c, (self.num_embeddings, self.embedding_dim), "float32"
        )
        return None
