# Extending Ember

Ember is designed so that the most common extension — **adding a new
element-wise tensor operator** — touches as few places as possible.

## Architecture at a glance

```
ember/                     Python front-end
  tensor/                  Tensor class, functional ops (sin, sum, ...), zeros/ones/full
  nn/                      layers (Linear, LayerNorm, MultiHeadAttention, ...)
  loss/, optim/            losses, optimizers
  random/                  seeding and random tensors
  data/                    Dataset, samplers, DataLoader
  cuda/                    device control, CUDA-graph capture
  checkpoint.py            save / load
  _core/                   compiled extension module (_tensor) + .pyi stubs

src/                       C / CUDA back-end
  core/                    memory, stream and cuBLAS handle (CPU and GPU variants)
  tensor/                  everything below links into the one _tensor module
    operators.def          <-- single source of truth for element-wise ops
    operators.h            declarations   (generated from operators.def)
    operators_cpu.c        CPU kernels    (generated from operators.def)
    operators_gpu.cu       CUDA kernels   (generated from operators.def)
    transformer.h          softmax, LayerNorm, bmm, embedding, attention
    transformer_*.c/.cu    their kernels; flash_*.c/.cu: fused attention
    tensor_module.h        what the binding files share (_TensorType, alloc_result)
    tensor_helpers.h       argument checks for the bindings
    tensor_api.c           _Tensor type, module init, element-wise + core bindings
    transformer_api.c      bindings for transformer.h
    random_api.c           random / constant fills
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

   importing `_abs` from `ember._core` (add it to the import list in
   `ember/_core/__init__.py`), and re-export it from `ember/tensor/__init__.py`
   and `ember/__init__.py`.

3. **Add the type stub** in `ember/_core/_tensor.pyi`:

   ```python
   def _abs(a: _Tensor) -> _Tensor: ...
   ```

4. **Add a test** in `tests/tensor/test_tensor.py` (the parametrized suite makes
   this a one-line addition to the `UNARY_OPS` list).

Rebuild (C changes need a reinstall; Python changes are live) and test:

```bash
uv pip install -e .
uv run --group dev pytest
```

## Adding a layer, optimizer, or loss

Layers subclass `ember.nn.base.Layer` and implement `forward` and `backward`.
A layer that owns parameter tensors also implements `parameters` and
`gradients`; a layer built from other layers implements `sublayers` instead
(see the `Layer` docstring). Optimizers subclass
`ember.optim.base.Optimizer` and implement `apply`. Losses subclass
`ember.loss.base.Loss` and implement `forward` and `backward`. Because these
are pure Python built on `Tensor`, no C changes are required.

## Non-element-wise operators

Operators that are not purely element-wise are written by hand for each
backend: the core ones (`matmul`, `transpose`, `sum`, `sum_axis`, the optimizer
steps) in `operators_cpu.c` / `operators_gpu.cu`, declared in `operators.h`
and bound in `tensor_api.c`; the transformer building blocks in
`transformer_*` / `flash_*`, declared in `transformer.h` and bound in
`transformer_api.c`. Follow the existing functions as templates.

A binding receives shapes as plain ints from Python and the kernels trust
them, so check every size it will index before launching (`require_size`,
`checked_size` and `parse_axis` in `tensor_helpers.h`); skip the kernel when
the result is empty. New device code belongs in the `_tensor` module: a second
extension module would link its own copy of the allocator and stream.
