"""Train a small decoder-only (GPT-style) char-level language model.

Downloads Tiny Shakespeare (~1.1 MB, Shakespeare's complete works as plain
text) on first run, batches fixed-length windows with
``ember.data.DataLoader``, and trains a causal transformer:

    Embedding -> PositionalEncoding -> TransformerEncoder(causal) -> Linear head

with AdamW. Writes ``losses.csv`` (train/val loss per eval step) and a
generated ``sample.txt`` next to this script.

Run with (from the repo root):

    uv run python examples/tiny-gpt/train.py
"""

from __future__ import annotations

import argparse
import csv
import time
import urllib.request
from functools import partial
from pathlib import Path

import numpy as np

import ember as em
import ember.data as data
import ember.loss as loss
import ember.nn as nn
import ember.optim as optim
from ember import Tensor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tinyshakespeare.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_data(path: Path = DATA_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATA_URL} -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path


class CharTokenizer:
    """Character-level tokenizer over the sorted set of unique characters."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[ch] for ch in text], dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class CharDataset(data.Dataset):
    """All ``block_size``-sized windows over a token array.

    Sample ``i`` is ``(tokens[i : i+block], tokens[i+1 : i+block+1])`` — the
    next-token input/target pair.
    """

    def __init__(self, tokens: np.ndarray, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        chunk = self.tokens[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def collate_lm(batch: list[tuple[np.ndarray, np.ndarray]], vocab: int):
    """Stack a batch into ``(B, S)`` int ids + ``(B*S, V)`` one-hot targets.

    The ids stay a NumPy integer array (``nn.Embedding`` consumes that
    directly); the targets become the float ``Tensor`` ``CrossEntropyLoss``
    expects.
    """
    xs = np.stack([b[0] for b in batch])
    ys = np.stack([b[1] for b in batch])
    b, s = xs.shape
    onehot = np.zeros((b * s, vocab), dtype=np.float32)
    onehot[np.arange(b * s), ys.reshape(-1)] = 1.0
    return xs, Tensor.from_np(onehot)


class GPT:
    """Decoder-only transformer language model built from ember's nn blocks."""

    def __init__(self, vocab: int, dim: int, heads: int, layers: int, block_size: int):
        self.vocab = vocab
        self.dim = dim
        self.block_size = block_size
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.PositionalEncoding(dim, max_len=block_size)
        self.blocks = nn.TransformerEncoder(
            layers, dim, heads, ff_hidden=4 * dim, causal=True, final_norm=True
        )
        self.head = nn.Linear(dim, vocab)

    def parameters(self) -> list[Tensor]:
        return (
            self.embed.parameters() + self.blocks.parameters() + self.head.parameters()
        )

    def gradients(self) -> list[Tensor | None]:
        return self.embed.gradients() + self.blocks.gradients() + self.head.gradients()

    def forward(self, idx: np.ndarray, training: bool = True) -> Tensor:
        self.batch, self.seq_len = idx.shape
        h = self.embed(idx)  # (B, S, dim)
        h = self.pos(h)
        h = self.blocks(h, training=training)
        h2 = Tensor._from_core(h._core, (self.batch * self.seq_len, self.dim), h.dtype)
        return self.head(h2, training=training)  # (B*S, vocab)

    def backward(self, grad_logits: Tensor) -> None:
        g = self.head.backward(grad_logits)
        g = Tensor._from_core(g._core, (self.batch, self.seq_len, self.dim), g.dtype)
        g = self.blocks.backward(g)
        g = self.pos.backward(g)
        self.embed.backward(g)

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int,
        temperature: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Autoregressively sample new tokens, cropping context to block_size."""
        idx = prompt_ids.reshape(1, -1).astype(np.int32)
        for _ in range(max_new_tokens):
            crop = idx[:, -self.block_size :]
            logits = self.forward(crop, training=False)
            last = logits.to_np()[-1]  # (V,) logits of the final position
            logits_t = last / max(temperature, 1e-6)
            probs = np.exp(logits_t - logits_t.max())
            probs /= probs.sum()
            nxt = rng.choice(self.vocab, p=probs)
            idx = np.concatenate([idx, [[nxt]]], axis=1)
        return idx[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # ~1750-2000 steps is the val-loss sweet spot for the default model; much
    # longer runs overfit (val loss climbs back up) since there is no dropout.
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--sample-tokens", type=int, default=500)
    p.add_argument(
        "--no-tf32",
        action="store_true",
        help="run matmuls in full fp32 instead of on the TF32 tensor cores",
    )
    args = p.parse_args()

    # TF32 is the standard precision for transformer training and is ~1.5x
    # faster end to end here; --no-tf32 opts back into full fp32.
    em.cuda.set_matmul_tf32(not args.no_tf32)

    path = download_data()
    text = path.read_text()
    tokenizer = CharTokenizer(text)
    tokens = tokenizer.encode(text)
    n_train = int(0.9 * len(tokens))
    train_ds = CharDataset(tokens[:n_train], args.block_size)
    val_ds = CharDataset(tokens[n_train:], args.block_size)

    collate = partial(collate_lm, vocab=tokenizer.vocab_size)
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        collate_fn=collate,
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    print(
        f"data: {len(tokens):,} tokens | vocab {tokenizer.vocab_size} | "
        f"train {len(train_ds):,} windows | val {len(val_ds):,} windows"
    )

    em.random.seed(args.seed)
    model = GPT(
        tokenizer.vocab_size, args.dim, args.heads, args.layers, args.block_size
    )
    n_params = sum(np.prod(t.shape) for t in model.parameters())
    print(
        f"model: {n_params:,} params | dim {args.dim}, {args.heads} heads, "
        f"{args.layers} layers, block {args.block_size}"
    )

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = loss.CrossEntropyLoss()

    print(f"training for {args.steps} steps (batch {args.batch_size}) ...")
    history: list[tuple[int, float, float]] = []
    t0 = time.perf_counter()
    train_iter = iter(train_loader)
    for step in range(args.steps):
        try:
            xb, yb = next(train_iter)
        except StopIteration:  # new epoch: reshuffle
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        logits = model.forward(xb, training=True)
        log_this_step = step % args.eval_interval == 0 or step == args.steps - 1
        if log_this_step:
            train_loss = crit(logits, yb)
            grad = crit.backward()
        else:
            grad = crit.gradient(logits, yb)  # no device->host sync
        model.backward(grad)
        opt.apply(model.gradients())

        if log_this_step:
            # validation loss over a few batches
            val_losses = []
            for i, (xv, yv) in enumerate(val_loader):
                if i >= args.eval_batches:
                    break
                val_losses.append(crit(model.forward(xv, training=False), yv))
            val_loss = float(np.mean(val_losses))
            history.append((step, train_loss, val_loss))
            dt = time.perf_counter() - t0
            tok_s = (step + 1) * args.batch_size * args.block_size / dt
            print(
                f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} "
                f"| ppl {np.exp(val_loss):.2f} | {tok_s:,.0f} tok/s"
            )

    dt = time.perf_counter() - t0
    print(f"done in {dt:.1f}s")

    with (BASE_DIR / "losses.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"])
        writer.writerows(history)
    print(f"loss history -> {BASE_DIR / 'losses.csv'}")

    rng = np.random.default_rng(args.seed)
    prompt = tokenizer.encode("\nKING:")
    sample = model.generate(prompt, args.sample_tokens, temperature=0.8, rng=rng)
    sample_text = tokenizer.decode(sample)
    (BASE_DIR / "sample.txt").write_text(sample_text)
    print(f"sample -> {BASE_DIR / 'sample.txt'}")
    print("\n--- sample ---")
    print(sample_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
