Stuff that needs to be done:

Done recently:
- Single source of truth for element-wise operators (`src/tensor/operators.def`)
  generates declarations, CPU kernels, CUDA kernels, and Python bindings via
  X-macros. Adding an operator is now a one-line change (see docs/extending.md).
- In-place operators (`+=`, `-=`, `*=`, `/=`) on `Tensor`.

- CI (build + pytest + ruff + mypy) across Python 3.11-3.14.
- Hardened C backend (cached cuBLAS handle, kernel launch error checks).
- softmax, log, amax, axis reductions with keepdims.
- N-D / batched matmul.
- Loss module (MSELoss, CrossEntropyLoss); SGD and AdamW optimizers.

Still to do:
- True in-place C kernels: `+=` etc. currently allocate a new buffer and adopt
  it. A dedicated in-place path would avoid the allocation.
- Further optimize the arithmetic kernels (fused ops, vectorization).
- Add JIT (for backend).
- Broaden dtype support (backend is float32-only today).
- Autograd (tape-based) to remove the hand-written layer/loss backward passes.
- Benchmarks and binary wheels (cibuildwheel) for the CPU build.
