Stuff that needs to be done:

Done recently:
- Single source of truth for element-wise operators (`src/tensor/operators.def`)
  generates declarations, CPU kernels, CUDA kernels, and Python bindings via
  X-macros. Adding an operator is now a one-line change (see docs/extending.md).
- In-place operators (`+=`, `-=`, `*=`, `/=`) on `Tensor`.

Still to do:
- True in-place C kernels: `+=` etc. currently allocate a new buffer and adopt
  it. A dedicated in-place path would avoid the allocation.
- Further optimize the arithmetic kernels (fused ops, vectorization).
- Add JIT (for backend).
- Broaden dtype support (backend is float32-only today).
- N-D matmul / batched matmul (currently 2-D only).
