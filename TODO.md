# Ember TODO

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
