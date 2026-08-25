# Data Loading

Ember ships a lightweight, PyTorch-style data pipeline in `ember.data`:
`Dataset`s yield individual samples, `Sampler`s decide the visit order, and a
`DataLoader` assembles batches.

Batch assembly happens in NumPy (`np.stack`) and floating-point batches are
converted to `Tensor` with a single backend copy — the per-batch work already
runs in C/NumPy, so the loader is single-process by design and adds no Python
overhead beyond index bookkeeping.

## Quick start

```python
import numpy as np

import ember.data as data
import ember.loss as loss
import ember.nn as nn
import ember.optim as optim

x = np.random.randn(512, 8).astype(np.float32)
y = np.random.randn(512, 1).astype(np.float32)

dataset = data.TensorDataset(x, y)
train_ds, val_ds = data.random_split(dataset, [410, 102], seed=0)

train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True, seed=0)

model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
opt = optim.Adam(model.parameters(), lr=1e-2)
criterion = loss.MSELoss()

for epoch in range(50):
    for xb, yb in train_loader:  # xb, yb are Tensors of shape (32, 8), (32, 1)
        pred = model(xb, training=True)
        criterion(pred, yb)
        model.backward(criterion.backward())
        opt.apply(model.gradients())
```

See `examples/dataloader.py` for the complete script.

## Datasets

| Class | Purpose |
| --- | --- |
| `Dataset` | Abstract base: implement `__getitem__` / `__len__`. `ds1 + ds2` concatenates. |
| `TensorDataset(*arrays)` | Wraps equal-length NumPy arrays / `Tensor`s; `ds[i]` returns the i-th rows as a tuple. |
| `Subset(dataset, indices)` | A view restricted to the given indices. |
| `ConcatDataset(datasets)` | Concatenates several datasets. |
| `random_split(dataset, lengths, seed=None)` | Random disjoint split (train/val). |

A custom dataset is just:

```python
class MyDataset(data.Dataset):
    def __init__(self, path): ...
    def __len__(self): ...
    def __getitem__(self, index):
        # return NumPy arrays / scalars; collation happens in the loader
        return x_i, y_i
```

## DataLoader

```python
data.DataLoader(
    dataset,
    batch_size=1,      # samples per batch
    shuffle=False,     # reshuffle every epoch
    sampler=None,      # custom index sampler (mutually exclusive with shuffle)
    batch_sampler=None,  # custom batch sampler (mutually exclusive with the above + drop_last)
    drop_last=False,   # drop the final incomplete batch
    collate_fn=None,   # custom batch assembly
    seed=None,         # reproducible shuffling (requires shuffle=True)
)
```

Iterating a `DataLoader` yields one collated batch per step; `len(loader)` is
the number of batches per epoch.

## Collation rules

`default_collate` stacks samples along a new leading batch axis:

- floating-point NumPy arrays / `Tensor`s / Python floats -> `Tensor`
  (the backend is float32-only),
- integer / boolean arrays and ints -> contiguous NumPy arrays (layers that
  consume integer ids, e.g. `nn.Embedding`, take NumPy input directly),
- tuples/lists are collated field-wise, dicts key-wise; strings pass through.

Pass `collate_fn=` to `DataLoader` for anything custom.

## Samplers

`SequentialSampler`, `RandomSampler(replacement=False, num_samples=None,
seed=None)`, and `BatchSampler(sampler, batch_size, drop_last)`. A custom
sampler only needs `__iter__` (yielding indices) and `__len__`:

```python
class ReversedSampler(data.Sampler):
    def __init__(self, n): self.n = n
    def __iter__(self): return iter(range(self.n - 1, -1, -1))
    def __len__(self): return self.n

loader = data.DataLoader(ds, batch_size=8, sampler=ReversedSampler(len(ds)))
```

## Fixed-shape batches and CUDA graphs

CUDA-graph capture requires fixed buffer addresses and shapes (see
[Performance](performance.md)). Use `drop_last=True` so every batch has the
same shape, allocate the batch tensors once, and refill them in place with
`Tensor.copy_from_numpy` before replaying:

```python
loader = data.DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, seed=0)
x_fixed = Tensor(np.zeros((32, in_dim), dtype=np.float32))
# ... capture graph on x_fixed ...
for xb, yb in loader:
    x_fixed.copy_from_numpy(xb.to_np())
    # graph.replay()
```
