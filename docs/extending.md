# Extending Ember

Ember is designed so that the most common extension — **adding a new
element-wise tensor operator** — touches as few places as possible.

## Architecture at a glance

```
ember/                     Python front-end
  tensor/tensor.py         Tensor class + functional ops (sin, sum, ...)
  nn/, optim/, random/     layers, optimizers, samplers
  _core/                   compiled extension modules (_tensor, _em_random) + .pyi stubs

src/                       C / CUDA back-end
  core/                    memory + type definitions (CPU and GPU variants)
  tensor/
    operators.def          <-- single source of truth for element-wise ops
    operators.h            declarations   (generated from operators.def)
    operators_cpu.c        CPU kernels    (generated from operators.def)
    operators_gpu.cu       CUDA kernels   (generated from operators.def)
    tensor_api.c           Python bindings (generated from operators.def)
  random/                  distributions (CPU + GPU)
```

The CPU (`*_cpu.c`) and GPU (`*_gpu.cu`) files provide the same functions; CMake
compiles exactly one set depending on whether a CUDA toolkit is found, so the
rest of the library is backend-agnostic.

## Adding an element-wise operator

All element-wise operators live in a single table,
[`src/tensor/operators.def`](https://github.com/pawel002/ember/blob/main/src/tensor/operators.def).
Each line is expanded by X-macros into the C declaration, the CPU
implementation, the CUDA kernel, and the Python binding. The expression must be
valid in both C and CUDA (`fmaxf`, `powf`, `expf`, … exist in both).

To add, say, an `abs` unary operator:

1. **Add one line to `operators.def`:**

   ```c
   EMBER_UNARY_OP(abs, fabsf(a[i]))
   ```

   (Use `EMBER_BINARY_OP`, `EMBER_SCALAR_OP`, or `EMBER_BROADCAST_OP` for the
   other kinds.) This alone gives you the `abs_tensor` kernel on both backends
   and the `_abs` function on the `ember._core._tensor` module.

2. **Expose it in Python** — add a thin wrapper in
   `ember/tensor/tensor.py`:

   ```python
   def abs(a: Tensor) -> Tensor:
       return _unary_op_wrapper(a, "abs()", _abs)
   ```

   and re-export it from `ember/tensor/__init__.py` and `ember/__init__.py`.

3. **Add the type stub** in `ember/_core/_tensor.pyi`:

   ```python
   def _abs(a: _Tensor) -> _Tensor: ...
   ```

4. **Add a test** in `tests/tensor/test_tensor.py` (the parametrized suite makes
   this a one-line addition to the `UNARY_OPS` list).

Rebuild and test:

```bash
uv pip install -e .
uv run --group dev pytest
```

## Adding a layer or optimizer

Layers subclass `ember.nn.base.Layer` and implement `forward`, `backward`,
`parameters`, `gradients`, and `reset`. Optimizers subclass
`ember.optim.base.Optimizer` and implement `apply`. Because these are pure
Python built on `Tensor`, no C changes are required.

## Non-element-wise operators

Operators that are not purely element-wise (`matmul`, `transpose`, `sum`,
`sum_axis`) are written by hand in `operators_cpu.c` / `operators_gpu.cu` and
declared in `operators.h`, then bound in `tensor_api.c`. Follow the existing
functions as templates.
