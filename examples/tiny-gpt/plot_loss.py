"""Plot the training/validation loss curves from ``losses.csv``.

Run with (from the repo root):

    uv run --with matplotlib python examples/tiny-gpt/plot_loss.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    with (BASE_DIR / "losses.csv").open() as f:
        rows = list(csv.DictReader(f))

    steps = [int(r["step"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train, label="train loss")
    ax.plot(steps, val, label="val loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("tiny-gpt on Tiny Shakespeare (char-level)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = BASE_DIR / "loss_curve.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
