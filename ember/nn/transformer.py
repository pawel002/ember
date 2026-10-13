import math
from collections.abc import Sequence

import numpy as np

from ember.tensor import Tensor

from .activations import GELU, ReLU
from .attention import MultiHeadAttention
from .base import Layer
from .layers import Linear
from .normalization import LayerNorm

_ACTIVATIONS: dict[str, type[Layer]] = {"gelu": GELU, "relu": ReLU}


class FeedForward(Layer):
    """Position-wise feed-forward network: Linear -> activation -> Linear.

    Applied independently to every position of a ``(..., dim)`` input, e.g.
    ``(batch, seq, dim)``. ``hidden`` defaults to ``4 * dim`` (the usual transformer expansion).
    """

    def __init__(self, dim: int, hidden: int | None = None, activation: str = "gelu"):
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; expected one of "
                f"{sorted(_ACTIVATIONS)}"
            )
        self.dim = dim
        self.hidden = hidden if hidden is not None else 4 * dim
        self.activation = activation

        self.fc1 = Linear(dim, self.hidden)
        self.act: Layer = _ACTIVATIONS[activation]()
        self.fc2 = Linear(self.hidden, dim)

    def sublayers(self) -> Sequence[Layer]:
        return (self.fc1, self.fc2)

    def forward(self, x: Tensor, training: bool) -> Tensor:
        h = self.fc1(x, training)
        h = self.act(h, training)
        return self.fc2(h, training)

    def backward(self, grad_y: Tensor) -> Tensor:
        g = self.fc2.backward(grad_y)
        g = self.act.backward(g)
        return self.fc1.backward(g)


class PositionalEncoding(Layer):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017), added to the
    input. Has no learnable parameters, so backward is the identity."""

    def __init__(self, dim: int, max_len: int = 5000):
        self.dim = dim
        self.max_len = max_len
        pe = np.zeros((max_len, dim), dtype=np.float32)
        pos = np.arange(max_len, dtype=np.float32)[:, None]
        div = np.exp(
            np.arange(0, dim, 2, dtype=np.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[: dim // 2])  # one column fewer if dim is odd
        self._pe_np = pe
        # one device copy per sequence length (stable buffers for CUDA graphs)
        self._cache: dict[int, Tensor] = {}

    def _pe(self, seq: int) -> Tensor:
        if seq > self.max_len:
            raise ValueError(f"sequence length {seq} exceeds max_len {self.max_len}")
        if seq not in self._cache:
            self._cache[seq] = Tensor.from_np(self._pe_np[:seq])
        return self._cache[seq]

    def forward(self, x: Tensor, training: bool) -> Tensor:
        # x: (batch, seq, dim); pe (seq, dim) broadcasts over the batch axis.
        return x + self._pe(x.shape[-2])

    def backward(self, grad_y: Tensor) -> Tensor:
        return grad_y


class TransformerEncoderLayer(Layer):
    """A pre-norm transformer encoder block:

        h   = x + SelfAttention(LayerNorm(x))
        out = h + FeedForward(LayerNorm(h))

    Pre-norm (LayerNorm before each sub-layer, residual around it) is the
    stable, widely-used arrangement. Set ``causal=True`` for decoder-style
    (autoregressive) masking.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden: int | None = None,
        causal: bool = False,
        activation: str = "gelu",
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden = ff_hidden if ff_hidden is not None else 4 * embed_dim
        self.causal = causal
        self.activation = activation

        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, causal)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, self.ff_hidden, activation)

    def sublayers(self) -> Sequence[Layer]:
        return (self.ln1, self.attn, self.ln2, self.ffn)

    def forward(self, x: Tensor, training: bool) -> Tensor:
        h = x + self.attn(self.ln1(x, training), training)
        return h + self.ffn(self.ln2(h, training), training)

    def backward(self, grad_y: Tensor) -> Tensor:
        # out = h + f, with f = ffn(ln2(h)): grad flows to h both directly and
        # through the ffn/ln2 branch.
        dn2 = self.ffn.backward(grad_y)
        dh = grad_y + self.ln2.backward(dn2)
        # h = x + a, with a = attn(ln1(x)): same split for x.
        dn1 = self.attn.backward(dh)
        dx = dh + self.ln1.backward(dn1)
        return dx


class TransformerEncoder(Layer):
    """A stack of :class:`TransformerEncoderLayer` blocks, with an optional final
    LayerNorm (standard for pre-norm stacks)."""

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_hidden: int | None = None,
        causal: bool = False,
        activation: str = "gelu",
        final_norm: bool = True,
    ):
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden = ff_hidden
        self.causal = causal
        self.activation = activation
        self.final_norm = final_norm

        self.layers = [
            TransformerEncoderLayer(embed_dim, num_heads, ff_hidden, causal, activation)
            for _ in range(num_layers)
        ]
        self.final_ln = LayerNorm(embed_dim) if final_norm else None

    def sublayers(self) -> Sequence[Layer]:
        if self.final_ln is None:
            return self.layers
        return [*self.layers, self.final_ln]

    def forward(self, x: Tensor, training: bool) -> Tensor:
        for layer in self.layers:
            x = layer(x, training)
        if self.final_ln is not None:
            x = self.final_ln(x, training)
        return x

    def backward(self, grad_y: Tensor) -> Tensor:
        if self.final_ln is not None:
            grad_y = self.final_ln.backward(grad_y)
        for layer in reversed(self.layers):
            grad_y = layer.backward(grad_y)
        return grad_y
