#ifndef TENSOR_TYPES_H
#define TENSOR_TYPES_H

#include <Python.h>

typedef struct {
    PyObject_HEAD void *d_ptr;
    int size;
} _Tensor;

#endif
