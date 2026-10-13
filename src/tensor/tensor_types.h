#ifndef TENSOR_TYPES_H
#define TENSOR_TYPES_H

#include <Python.h>

/* The object behind ember._core._tensor._Tensor: a flat float32 buffer on the
 * active backend (device memory on CUDA). Empty tensors have d_ptr == NULL. */
typedef struct {
    PyObject_HEAD
    void *d_ptr;
    int size;
} _Tensor;

#endif
