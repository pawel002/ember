"""Tiny GPT-style transformer trained to memorize a copy task.

Shows the transformer building blocks composing into a full model with a
hand-written forward/backward and an Adam training loop:

    Embedding -> PositionalEncoding -> TransformerEncoder(causal) -> Linear head

Run with:

    export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
    uv run python examples/transformer.py
"""

from __future__ import annotations

import math

import numpy as np

import ember as em
import ember.loss as loss
import ember.nn as nn
import ember.optim as optim
from ember import Tensor


class TinyGPT:
    """A minimal causal language model built from ember's nn blocks."""

    def __init__(self, vocab: int, dim: int, heads: int, layers: int, seq: int):
        self.vocab = vocab
        self.dim = dim
        self.seq = seq
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.PositionalEncoding(dim, max_len=seq)
        self.encoder = nn.TransformerEncoder(
            layers, dim, heads, ff_hidden=4 * dim, causal=True, final_norm=True
        )
        self.head = nn.Linear(dim, vocab)  # project each position to logits

    def parameters(self) -> list[Tensor]:
        return (
            self.embed.parameters() + self.encoder.parameters() + self.head.parameters()
        )

    def gradients(self) -> list[Tensor | None]:
        return self.embed.gradients() + self.encoder.gradients() + self.head.gradients()

    def forward(self, idx: np.ndarray) -> Tensor:
        self.batch, self.seq_len = idx.shape
        h = self.embed(idx)  # (B, S, dim)
        h = self.pos(h)  # + positional encoding
        h = self.encoder(h, training=True)  # (B, S, dim)
        h2 = Tensor._from_core(h._core, (self.batch * self.seq_len, self.dim), h.dtype)
        logits = self.head(h2, training=True)  # (B*S, vocab)
        return logits

    def backward(self, grad_logits: Tensor) -> None:
        g = self.head.backward(grad_logits)  # (B*S, dim)
        g = Tensor._from_core(g._core, (self.batch, self.seq_len, self.dim), g.dtype)
        g = self.encoder.backward(g)
        g = self.pos.backward(g)
        self.embed.backward(g)  # populates embedding grad


def main() -> int:
    em.random.seed(0)
    rng = np.random.default_rng(0)

    vocab, dim, heads, layers = 16, 32, 4, 2
    batch, seq = 8, 12

    # Fixed random token sequences; the model learns to predict the next token.
    tokens = rng.integers(0, vocab, size=(batch, seq + 1)).astype(np.int32)
    inp = tokens[:, :-1]
    tgt = tokens[:, 1:]
    # one-hot targets of shape (B*S, vocab) for CrossEntropyLoss
    tgt_onehot = np.zeros((batch * seq, vocab), dtype=np.float32)
    tgt_onehot[np.arange(batch * seq), tgt.reshape(-1)] = 1.0
    target = Tensor.from_np(tgt_onehot)

    model = TinyGPT(vocab, dim, heads, layers, seq)
    opt = optim.Adam(model.parameters(), lr=3e-3)
    crit = loss.CrossEntropyLoss()

    for step in range(400):
        logits = model.forward(inp)
        loss_val = crit(logits, target)
        model.backward(crit.backward())
        opt.apply(model.gradients())
        if step % 50 == 0 or step == 399:
            preds = logits.to_np().reshape(batch, seq, vocab).argmax(-1)
            acc = (preds == tgt).mean()
            print(f"step {step:4d}  loss {loss_val:.4f}  next-token acc {acc:.3f}")

    print(f"\nperplexity: {math.exp(loss_val):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
