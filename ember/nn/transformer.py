import math

import numpy as np

from ember.tensor import Tensor

from ._functional import view
from .activations import GELU, ReLU
from .attention import MultiHeadAttention
from .base import Layer
from .layers import Linear
from .normalization import LayerNorm


class FeedForward(Layer):
    """Position-wise feed-forward network: Linear -> activation -> Linear.

    Applied independently to every position of a ``(batch, seq, dim)`` input
    (the two Linear layers act on the flattened ``(batch*seq, dim)`` view).
    ``hidden`` defaults to ``4 * dim`` (the usual transformer expansion).
    """

    def __init__(self, dim: int, hidden: int | None = None, activation: str = "gelu"):
        self.dim = dim
        self.hidden = hidden if hidden is not None else 4 * dim
        self.activation = activation
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        self._shape: tuple[int, ...] | None = None
        self.fc1 = Linear(self.dim, self.hidden)
        self.act: Layer = GELU() if self.activation == "gelu" else ReLU()
        self.fc2 = Linear(self.hidden, self.dim)

    def parameters(self) -> list[Tensor]:
        return self.fc1.parameters() + self.fc2.parameters()

    def gradients(self) -> list[Tensor | None]:
        return self.fc1.gradients() + self.fc2.gradients()

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        self._shape = x.shape
        n = math.prod(x.shape[:-1])
        h = view(x, (n, self.dim))
        h = self.fc1(h, training)
        h = self.act(h, training)
        h = self.fc2(h, training)
        self.y = view(h, x.shape)
        return self.y

    def backward(self, grad_y: Tensor) -> Tensor:
        assert self._shape is not None, "forward() must run before backward()"
        n = math.prod(self._shape[:-1])
        g = view(grad_y, (n, self.dim))
        g = self.fc2.backward(g)
        g = self.act.backward(g)
        g = self.fc1.backward(g)
        return view(g, self._shape)


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
        pe[:, 1::2] = np.cos(pos * div)
        self._pe_np = pe
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        self._cache: dict[int, Tensor] = {}

    def parameters(self) -> list[Tensor]:
        return []

    def gradients(self) -> list[Tensor | None]:
        return []

    def _pe(self, seq: int) -> Tensor:
        assert seq <= self.max_len, (
            f"sequence length {seq} exceeds max_len {self.max_len}"
        )
        if seq not in self._cache:
            self._cache[seq] = Tensor.from_np(self._pe_np[:seq])
        return self._cache[seq]

    def forward(self, x: Tensor, training: bool) -> Tensor:
        # x: (batch, seq, dim); pe (seq, dim) broadcasts over the batch axis.
        self.x = x
        seq = x.shape[-2]
        self.y = x + self._pe(seq)
        return self.y

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
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        self._h: Tensor | None = None
        self.ln1 = LayerNorm(self.embed_dim)
        self.attn = MultiHeadAttention(self.embed_dim, self.num_heads, self.causal)
        self.ln2 = LayerNorm(self.embed_dim)
        self.ffn = FeedForward(self.embed_dim, self.ff_hidden, self.activation)

    def _sublayers(self) -> tuple[Layer, ...]:
        return (self.ln1, self.attn, self.ln2, self.ffn)

    def parameters(self) -> list[Tensor]:
        out: list[Tensor] = []
        for layer in self._sublayers():
            out.extend(layer.parameters())
        return out

    def gradients(self) -> list[Tensor | None]:
        out: list[Tensor | None] = []
        for layer in self._sublayers():
            out.extend(layer.gradients())
        return out

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        a = self.attn(self.ln1(x, training), training)
        h = x + a
        f = self.ffn(self.ln2(h, training), training)
        self._h = h
        self.y = h + f
        return self.y

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
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        self.layers = [
            TransformerEncoderLayer(
                self.embed_dim,
                self.num_heads,
                self.ff_hidden,
                self.causal,
                self.activation,
            )
            for _ in range(self.num_layers)
        ]
        self.final_ln = LayerNorm(self.embed_dim) if self.final_norm else None

    def _sublayers(self) -> list[Layer]:
        layers: list[Layer] = list(self.layers)
        if self.final_ln is not None:
            layers.append(self.final_ln)
        return layers

    def parameters(self) -> list[Tensor]:
        out: list[Tensor] = []
        for layer in self._sublayers():
            out.extend(layer.parameters())
        return out

    def gradients(self) -> list[Tensor | None]:
        out: list[Tensor | None] = []
        for layer in self._sublayers():
            out.extend(layer.gradients())
        return out

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.x = x
        for layer in self.layers:
            x = layer(x, training)
        if self.final_ln is not None:
            x = self.final_ln(x, training)
        self.y = x
        return x

    def backward(self, grad_y: Tensor) -> Tensor:
        if self.final_ln is not None:
            grad_y = self.final_ln.backward(grad_y)
        for layer in reversed(self.layers):
            grad_y = layer.backward(grad_y)
        return grad_y
