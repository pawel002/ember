Stuff that needs to be done:
- True in-place C kernels: `+=` etc. currently allocate a new buffer and adopt
  it. A dedicated in-place path would avoid the allocation.
- Further optimize the arithmetic kernels (fused ops, vectorization).
- Add JIT (for backend).
- Broaden dtype support (backend is float32-only today).
