Stuff that needs to be done:
- True in-place C kernels: `+=` etc. currently allocate a new buffer and adopt
  it. A dedicated in-place path would avoid the allocation.
- Further optimize the arithmetic kernels (fused ops, vectorization).
- Add JIT (for backend).
- Broaden dtype support (backend is float32-only today).

Recommended order (by ROI)

0. Build a benchmark + profiler harness first. You cannot beat PyTorch without measuring against it. A tiny MLP on synthetic data, steps/sec vs torch eager and torch.compile, plus nsys traces. This tells you where the time actually goes and prevents optimizing the wrong thing. Do this before any perf work.

1. Caching device allocator (biggest immediate win, low risk). Replace per-op cudaMalloc/cudaFree with a pool: a free-list keyed by buffer size that hands back recycled device blocks. This is ~a day of work in memory_gpu.cu, removes the dominant cost, and — critically — gives stable pointers across iterations, which CUDA graphs require. This one change alone may get you close to PyTorch.

2. Eliminate mid-step host syncs. Replace the sum() host round-trip with a real on-device reduction kernel; fix the broadcast path to not cudaMalloc/memcpy/sync every call (pass shape/strides by value with a fixed max-ndim, or a persistent small buffer). Run everything on one stream and only synchronize when the user calls .to_np(). Now the step is fully asynchronous.

3. Kernel fusion for elementwise chains, starting with the optimizer. Collapse the whole Adam/AdamW update into one fused kernel (ideally one launch over all parameters), and fuse activation-and-derivative and the bias + x@w epilogue. This cuts op count (and therefore launches/allocs) by ~10× in the parts that aren't matmul. Fused optimizer kernels are a classic, large win.

4. CUDA Graph capture/replay of the full step. With 1–3 done (stable pointers, no host syncs, fewer ops), capture the step once and replay. This is the endgame that makes overhead vanish. Expose it as something like train_step = ember.capture(step_fn); for batch in data: train_step().

5. Supporting pieces: true in-place kernels (so += doesn't allocate), pinned host memory + async H2D for feeding batches, and keeping a single persistent stream.

The honest strategic take

- The prerequisite for fusion and graph capture is knowing the op sequence, i.e. a tiny trace/IR. You don't need a full autograd engine (your manual backward is fine), but a minimal "record the step's ops once, then plan memory + fuse + capture" layer is the real architectural investment — items 3 and 4 are its payoff.
- This wins specifically for small, fixed-shape models. Be upfront that dynamic shapes or large models erode the advantage (there you're FLOP-bound and just calling cuBLAS/cuDNN like everyone else). Your pitch is "tiny, fixed nets, zero-overhead replay, no 2GB framework" — lean into that.
- #1 (caching allocator) is what I'd build first after t ROI, self-contained, and a hard prerequisite for thegraph work.

If you want, I can start by building the benchmark harnesec vs a PyTorch baseline if it's installed) so we have anumber to beat, and then implement the caching allocator and measure the delta. Want me to go ahead with those two?