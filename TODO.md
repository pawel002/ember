# Ember TODO

## Done: performance (implemented, pending on-machine benchmark)

All eight points below are implemented. They were written while the dev host's
GPU driver was mid-upgrade (kernel 535 vs userspace 580), so they have NOT yet
been built/benchmarked here -- verify on a working machine (reboot first, then
`uv run --group bench python benchmarks/bench_mlp.py`).

1. **Fused activation kernels.** ReLU/Sigmoid/Tanh/GELU forward and backward are
   single kernels (generated from operators.def where they are unary/binary;
   GELU backward is hand-written).
2. **Fused Linear epilogue.** `matmul_bias` = GEMM + row-broadcast bias add,
   used by `Linear.forward` (no broadcast temporary).
3. **Custom tiny-shape matmul.** Below a work threshold, matmul uses a simple
   kernel instead of cuBLAS to dodge its launch overhead.
4. **Warp-shuffle reductions.** Full `sum` uses a warp-shuffle block reduction.
5. **Pinned host memory + async H2D.** `Tensor.copy_from_numpy` uploads through a
   pinned staging buffer via `cudaMemcpyAsync` on the ember stream.
6. **Graph ergonomics.** Batch feeding into a stable buffer before replay; opt-in
   `capturable=True` Adam/AdamW (device-side step counter + bias correction) that
   stays exact under CUDA-graph capture.
7. **Grouped (multi-tensor) optimizer.** Opt-in `Adam(foreach=True)` updates all
   parameters in one kernel launch.
8. **(prior) caching allocator, in-place kernels, fused optimizer, no mid-step
   host syncs, CUDA-graph capture** -- see git history.

### Remaining perf follow-ups
- Persistent per-op result buffers / static step-buffer planning (only the
  reduction path and graph replay avoid per-op allocation today).
- cublasLt GEMM+bias epilogue (fuse bias into the GEMM itself).
- Tiled/shared-memory small GEMM (current tiny-shape kernel is naive).
- Warp-shuffle `sum_axis`/`max_axis`; multi-stream overlap; grouped + capturable
  optimizer combined; `capture()` re-record on shape change.

## Next: functionality
- Autograd (tape-based) to remove the hand-written layer/loss backward passes.
- Broaden dtype support (backend is float32-only today).
- More layers (Conv, LayerNorm, Embedding) and a data-loading path.
- Benchmarks in CI (regression guard) and binary wheels (cibuildwheel) for the
  CPU build.
