"""Train a small MLP with ``ember.data`` on a synthetic regression task.

Mirrors the model from docs/usage.md, but data flows through a
``TensorDataset`` + ``DataLoader`` with shuffling, mini-batching and a
held-out validation split via ``random_split``.
"""

import numpy as np

import ember as em
import ember.data as data
import ember.loss as loss
import ember.nn as nn
import ember.optim as optim
from ember import Tensor

# synthetic dataset: y = x @ w_true + b_true + noise
rng = np.random.default_rng(42)
n, in_dim = 512, 8
w_true = rng.standard_normal((in_dim, 1)).astype(np.float32)
x = rng.standard_normal((n, in_dim)).astype(np.float32)
y = x @ w_true + 0.3 + 0.05 * rng.standard_normal((n, 1)).astype(np.float32)

dataset = data.TensorDataset(x, y)
train_ds, val_ds = data.random_split(dataset, [int(0.8 * n), n - int(0.8 * n)], seed=0)

train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True, seed=0)
val_loader = data.DataLoader(val_ds, batch_size=len(val_ds))

em.random.seed(0)  # reproducible weight init
model = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))
opt = optim.Adam(model.parameters(), lr=1e-2)
criterion = loss.MSELoss()

for epoch in range(50):
    model_loss = 0.0
    for xb, yb in train_loader:
        pred = model(xb, training=True)
        model_loss += criterion(pred, yb)
        model.backward(criterion.backward())
        opt.apply(model.gradients())

    if epoch % 10 == 0 or epoch == 49:
        xv, yv = next(iter(val_loader))
        val_loss = criterion(model(xv, training=False), yv)
        print(
            f"epoch {epoch:3d} | train {model_loss / len(train_loader):.5f} "
            f"| val {val_loss:.5f}"
        )

# The loader yields ``Tensor`` batches for float data, so they can also be
# fed into a fixed buffer captured by a CUDA graph (use drop_last=True):
xb, yb = next(iter(train_loader))
x_fixed = Tensor(np.zeros_like(xb.to_np()))
x_fixed.copy_from_numpy(xb.to_np())
print("fixed-buffer feed ok:", x_fixed.shape)
