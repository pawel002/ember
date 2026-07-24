# Ember TODO

## Done

### Library / API
- Single source of truth for element-wise operators (`src/tensor/operators.def`,
  X-macros generate declarations, CPU + CUDA kernels, and Python bindings).
- softmax, log, amax, axis reductions with keepdims.
- N-D / batched matmul.
- Loss module (MSELoss, CrossEntropyLoss); SGD, Adam, and AdamW optimizers.
- CI (build + pytest + ruff + mypy) across Python 3.11-3.14.

### Performance (see docs/performance.md, benchmarks/bench_mlp.py)
- True in-place elementwise kernels (`+=` mutates in place, stable addresses).
- Caching device allocator (recycle freed blocks by size; no per-op
  cudaMalloc/cudaFree).
- No mid-step host syncs (by-value broadcast metadata, on-device sum,
  `Loss.gradient()` avoids the scalar readback).
- Fused optimizer kernels (one in-place kernel per parameter).
- CUDA-graph capture/replay of the training step (`ember.cuda.capture`).
- Result on RTX 3090 vs PyTorch eager: eager Adam 2.8-8x; captured-graph SGD
  up to ~15x on small models.

## Next: performance

Roughly in priority order (biggest expected win first).

1. **Fuse the elementwise/activation graph, not just the optimizer.** The
   forward/backward still launch one kernel per op (e.g. sigmoid backward is
   `y*(1-y)*grad` = 3 launches + 2 temporaries). A small fused-elementwise path
   (either hand-written fused kernels for the activations, or a tiny expression
   fuser that JITs a chain of ops into one kernel) removes most remaining
   launches outside matmul. Fuses naturally with in-place output.
2. **Persistent scratch / no per-op result allocation.** Even with the caching
   allocator every op still allocates a result buffer and refcounts a Python
   object. Pre-plan the step's buffers once (static allocation for fixed shapes)
   so replay-outside-graph and first-iteration cost drop too.
3. **Pinned host memory + async H2D for input feeding.** Real training copies a
   new batch each step; `copy_to_device` is currently synchronous from pageable
   memory. Pinned staging buffers + `cudaMemcpyAsync` on the ember stream let
   the copy overlap and stay inside graph capture.
4. **Fused Linear epilogue.** `bias + x@w` is a GEMM followed by a separate
   broadcast-add kernel; use cuBLAS GEMM with bias epilogue (cublasLt) or a
   fused bias kernel to save a launch + a pass over the output.
5. **Better matmul for tiny shapes.** cuBLAS has fixed launch overhead that
   dominates for small matrices; a custom tiled kernel (or cublasLt with a
   tuned algo) can win at the sizes ember targets.
6. **Faster reductions.** `sum`/`sum_axis`/`max_axis` are simple; use warp
   shuffles / multiple elements per thread, and a two-pass tree reduction for
   full `sum` instead of a single atomic-add pass.
7. **Graph ergonomics.** Let `capture()` re-record when input shapes change,
   support feeding fresh batches into captured input buffers, and update Adam's
   bias-correction inside the captured step (currently frozen at capture time).
8. **Multi-stream / overlap.** For independent branches (e.g. per-parameter
   optimizer work) overlap on multiple streams; group all parameters into one
   fused optimizer launch instead of one-per-parameter.

## Next: functionality
- Autograd (tape-based) to remove the hand-written layer/loss backward passes.
- Broaden dtype support (backend is float32-only today).
- More layers (Conv, LayerNorm, Embedding) and a data-loading path.
- Benchmarks in CI (regression guard) and binary wheels (cibuildwheel) for the
  CPU build.
