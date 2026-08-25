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
- **Fused activation kernels** — ReLU/Sigmoid/Tanh/GELU forward and backward are
  one kernel each instead of a chain of elementwise ops.
- **Fused Linear epilogue** — `Linear.forward` runs GEMM + bias in `matmul_bias`
  with no broadcast temporary. `Linear.backward` passes its two transposed
  operands as cuBLAS `OP_T` flags rather than materializing transposed copies.
- **Fused ("flash") attention** — the `(seq, seq)` score matrix never reaches
  global memory: each block keeps a tile of queries resident, streams the
  key/value tiles past it, and carries the softmax as a running `(max, sum)`
  pair. Backward recomputes the scores from the saved log-sum-exp. The kernels
  also take the head interleaving as a stride, so they read and write the
  projections as `(batch, seq, heads, head_dim)` in place — the split-into-heads
  transpose (four copies forward, four back, per block) disappears entirely.
  Available for `head_dim` 16/32/64/128; other sizes fall back to the composed
  GEMM + softmax + GEMM path automatically.
- **Blocked reductions** — LayerNorm's backward accumulates `dgamma`/`dbeta`
  across a strip of rows in shared memory (rather than one global atomic per
  element), and axis reductions with few outputs but a long axis split into
  row-strips that combine in a second pass.
- **No mid-step host syncs** — broadcasts pass metadata by value, reductions run
  on device (full `sum` via a warp-shuffle reduction), and `Loss.gradient()`
  computes the gradient without reading the scalar loss back to the host.
- **Small-matmul fast path** — below a size threshold matmul uses a lightweight
  kernel instead of paying cuBLAS's fixed launch overhead.
- **Pinned async uploads** — `Tensor.copy_from_numpy` stages through pinned
  memory with an async copy, so fresh batches can be fed into a stable buffer.
- **CUDA graphs** — capture the whole training step once and replay it, removing
  nearly all launch and Python-dispatch overhead.

### Matmul precision

`ember.cuda.set_matmul_tf32(True)` runs every cuBLAS GEMM on the TF32 tensor
cores instead of the fp32 pipeline — roughly 3x the GEMM throughput on Ampere
and later, for a 10-bit input mantissa with fp32 accumulation. This is the same
trade as PyTorch's `torch.backends.cuda.matmul.allow_tf32`, and like PyTorch it
is **off by default** because it is a real precision change. It is worth ~1.5x
end-to-end on the tiny-gpt example with no visible effect on the loss curve.

### Optimizer options for the hot loop

- `Adam(..., foreach=True)` / `AdamW(..., foreach=True)` update all parameters
  in one grouped kernel launch instead of one per parameter (a ~100-parameter
  transformer goes from 100 launches per step to 1). The per-parameter device
  pointer arrays are uploaded once and then reused, so the steady state costs a
  single launch with no host<->device traffic.
- `Adam(..., capturable=True)` / `AdamW(..., capturable=True)` keep the step
  counter and bias correction on-device so Adam stays numerically exact when the
  training step is captured into a CUDA graph (SGD is already exact under
  capture). Feed new batches with `x.copy_from_numpy(batch)` before `replay()`.

## Benchmark

`benchmarks/bench_mlp.py` compares ember with PyTorch (eager) on small MLPs:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
uv run --group bench python benchmarks/bench_mlp.py
```

On an RTX 3090, ember beats PyTorch eager on every configuration — by ~2.8x at
the low end for eager Adam training, and up to ~15x for a small model using a
captured CUDA graph.

### Transformer training (tiny-gpt)

`examples/tiny-gpt/train.py` (6 layers, dim 256, 8 heads, block 256, batch 64)
on an RTX 6000 Ada, measured per training step:

| | ms/step | tok/s | 751 steps |
|---|---|---|---|
| before these kernels (composed attention, fp32) | 78.1 | 210k | 59.3 s |
| after, full fp32 (`--no-tf32`) | 31.0 | 528k | 25.4 s |
| after, TF32 (the example's default) | 18.7 | 876k | 15.4 s |

The loss curve is unchanged across all three (val loss after 751 steps:
1.66 / 1.65 / 1.57 — within run-to-run noise).

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
