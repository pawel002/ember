# Ember TODO

### Remaining perf follow-ups
- Persistent per-op result buffers / static step-buffer planning (only the
  reduction path and graph replay avoid per-op allocation today).
- cublasLt epilogues: fuse bias into the GEMM (`EPILOGUE_BIAS`), GELU into the
  FFN's first GEMM (`GELU_AUX_BIAS`), and the bias gradient into the backward
  GEMM (`BGRADA`) -- together ~1.5 ms/step on tiny-gpt.
- Tiled/shared-memory small GEMM (current tiny-shape kernel is naive).
- `max_axis` still uses the naive per-output kernel (`sum_axis` no longer does).
- Vectorized (float4) elementwise + LayerNorm forward kernels.
- Multi-stream overlap; grouped + capturable optimizer combined; `capture()`
  re-record on shape change.
- `Embedding` forward/backward still upload indices with a blocking copy and a
  `sync_device()` per call, which drains the pipeline twice a step and blocks
  CUDA-graph capture of any model with an embedding.
- Fused cross-entropy taking int32 targets (kills the host-side one-hot build
  and its 4.3 MB upload per batch in tiny-gpt).
- Flash attention runs at ~15 TFLOPS dense-equivalent; a wider register tile in
  the dK/dV accumulation, or a single backward kernel with atomic `dQ`
  accumulation (5 score recomputes instead of 7), would push it further.
- `add_broadcasted` runs at ~250 GB/s (PositionalEncoding's `x + pe`); a
  contiguous-tail fast path would fix it.

## Next: functionality
- Autograd (tape-based) to remove the hand-written layer/loss backward passes.
- Broaden dtype support (backend is float32-only today).
- Cross-attention (separate q / kv inputs) -- the fused attention kernels
  already take separate `sq`/`sk`, only the layer assumes self-attention.
- Fused attention for head_dims outside {16, 32, 64, 128} (those fall back to
  the composed path).
- More layers (Conv).
- Benchmarks in CI (regression guard) and binary wheels (cibuildwheel) for the
  CPU build.
  