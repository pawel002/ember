# Performance

Ember targets **small, fixed-shape networks**, where training time is dominated
by per-operation overhead (allocation, kernel launches, host synchronization)
rather than raw compute. Ember removes that overhead so it can out-run a full
framework in this regime.

## What makes it fast

- **Caching device allocator** — freed device buffers are recycled by size, so
  a steady-state training loop never calls `cudaMalloc`/`cudaFree`.
- **In-place ops and fused optimizers** — `+=` mutates in place, and each
  Adam/AdamW/SGD update is a single fused kernel instead of ~10 ops.
- **No mid-step host syncs** — broadcasts pass metadata by value, reductions run
  on device, and `Loss.gradient()` computes the gradient without reading the
  scalar loss back to the host.
- **CUDA graphs** — capture the whole training step once and replay it, removing
  nearly all launch and Python-dispatch overhead.

## Benchmark

`benchmarks/bench_mlp.py` compares ember with PyTorch (eager) on small MLPs:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
uv run --group bench python benchmarks/bench_mlp.py
```

On an RTX 3090, ember beats PyTorch eager on every configuration — by ~2.8x at
the low end for eager Adam training, and up to ~15x for a small model using a
captured CUDA graph.

## CUDA graphs (`ember.cuda`)

For a fixed-shape training step, capture it once and replay:

```python
import ember as em
import ember.nn as nn
import ember.optim as optim
import ember.loss as loss

model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
opt = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
criterion = loss.MSELoss()

def step():
    pred = model(x, training=True)             # x, target are fixed buffers
    model.backward(criterion.gradient(pred, target))
    opt.apply(model.gradients())

graph = em.cuda.capture(step)   # warms up, then records one step
for _ in range(n_steps):
    graph.replay()              # near-zero overhead
em.cuda.sync()                  # block for results
```

**Requirements / caveats:**

- Fixed shapes; feed new data by copying into the same input buffers.
- Use `Loss.gradient()` (no scalar read) in the captured step — a host sync
  cannot occur during capture.
- SGD is exact under capture. Adam/AdamW freeze their bias-correction factor at
  capture time; capture after enough warmup steps (when the correction is
  ~1) or prefer SGD if you need it exact.
- On the CPU backend `ember.cuda` is a no-op.
