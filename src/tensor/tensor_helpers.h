/* Small host-side helpers for the binding files. */
#ifndef EMBER_TENSOR_HELPERS_H
#define EMBER_TENSOR_HELPERS_H

#include <Python.h>
#include <limits.h>
#include <stdlib.h>

#include "../core/tensor_types.h"

static inline PyObject *build_nested_list(float *data, long *dims, int ndim, int current_dim,
                                          int *offset)
{
    long size = dims[current_dim];
    PyObject *list = PyList_New(size);
    if (!list) return NULL;

    if (current_dim == ndim - 1) {
        for (long i = 0; i < size; i++) {
            PyObject *num = PyFloat_FromDouble(data[*offset]);
            if (!num) {
                Py_DECREF(list);
                return NULL;
            }
            PyList_SetItem(list, i, num);
            (*offset)++;
        }
    }

    else {
        for (long i = 0; i < size; i++) {
            PyObject *sublist = build_nested_list(data, dims, ndim, current_dim + 1, offset);
            if (!sublist) {
                Py_DECREF(list);
                return NULL;
            }
            PyList_SetItem(list, i, sublist);
        }
    }

    return list;
}

/* Copy a tuple of non-negative ints into a malloc'd array. Returns 0, or -1
 * with a Python error set. */
static inline int tuple_to_array(PyObject *tuple, int **array_out, int *size_out)
{
    int size = (int)PyTuple_GET_SIZE(tuple);
    int *array = (int *)malloc((size ? size : 1) * sizeof(int));
    if (!array) {
        PyErr_NoMemory();
        return -1;
    }

    for (int i = 0; i < size; i++) {
        long v = PyLong_AsLong(PyTuple_GET_ITEM(tuple, i));
        if (v == -1 && PyErr_Occurred()) {
            free(array);
            return -1;
        }
        if (v < 0 || v > INT_MAX) {
            free(array);
            PyErr_Format(PyExc_ValueError, "invalid dimension %ld", v);
            return -1;
        }
        array[i] = (int)v;
    }

    *array_out = array;
    *size_out = size;
    return 0;
}

/* ---- argument validation ----
 * The bindings receive shapes as plain ints from Python, and the kernels trust
 * them, so every size a kernel will index is checked here first: a wrong shape
 * becomes a ValueError instead of an out-of-bounds device access. Sizes are
 * computed in long long so an overflowing product is caught too. */

/* `n` as an element count: non-negative and representable as int. Returns it,
 * or -1 with ValueError set. */
static inline int checked_size(long long n)
{
    if (n < 0 || n > INT_MAX) {
        PyErr_Format(PyExc_ValueError, "invalid tensor size %lld (must be 0..%d)", n, INT_MAX);
        return -1;
    }
    return (int)n;
}

/* 0 if `t` holds exactly `expect` elements, else -1 with ValueError set. */
static inline int require_size(const _Tensor *t, long long expect, const char *name)
{
    if (t->size != expect) {
        PyErr_Format(PyExc_ValueError, "%s has %d elements, expected %lld", name, t->size, expect);
        return -1;
    }
    return 0;
}

/* Describe a reduction over `axis` of a tensor with `shape_obj` and `numel`
 * elements as (outer, axis_dim, inner): `outer` positions before the axis,
 * `inner` after it. Returns 0, or -1 with a Python error set. */
static inline int parse_axis(PyObject *shape_obj, int axis, int numel, int *outer, int *axis_dim,
                             int *inner)
{
    int ndim, *shape;
    if (tuple_to_array(shape_obj, &shape, &ndim) < 0) return -1;
    if (axis < 0 || axis >= ndim) {
        free(shape);
        PyErr_Format(PyExc_ValueError, "axis %d out of range for %d dimensions", axis, ndim);
        return -1;
    }

    long long before = 1, after = 1;
    for (int i = 0; i < axis; i++) before *= shape[i];
    for (int i = axis + 1; i < ndim; i++) after *= shape[i];
    long long total = before * shape[axis] * after;
    *axis_dim = shape[axis];
    free(shape);

    if (total != numel) {
        PyErr_Format(PyExc_ValueError, "shape describes %lld elements, tensor has %d", total,
                     numel);
        return -1;
    }
    *outer = (int)before;
    *inner = (int)after;
    return 0;
}

#endif
