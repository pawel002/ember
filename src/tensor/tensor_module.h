/* Internal interface of the _tensor extension module, shared by its binding
 * translation units (tensor_api.c, transformer_api.c). They all link into the
 * one _tensor.so, so every op shares its CUDA stream and caching allocator. */
#ifndef EMBER_TENSOR_MODULE_H
#define EMBER_TENSOR_MODULE_H

#include <Python.h>

#include "../core/tensor_types.h"

#define OP_METHOD(NAME) {#NAME, (PyCFunction)NAME, METH_VARARGS, "Element-wise " #NAME " operation"}

/* The _Tensor Python type (defined in tensor_api.c). */
extern PyTypeObject _TensorType;

/* A new _Tensor with `size` uninitialized device floats, or NULL with a Python
 * error set. */
_Tensor *alloc_result(int size);

/* Method tables of the other binding files, added to the module at init. */
extern PyMethodDef transformer_methods[];

#endif
