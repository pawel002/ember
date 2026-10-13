/* Python bindings for random and constant fills. Part of the _tensor module (see
 * tensor_module.h), so the results come from the same caching allocator as
 * every other tensor.
 *
 * Values are generated on the host with the C stdlib rand(), so a given seed
 * produces the same numbers on both backends, and then copied to the device. */

#include <Python.h>
#include <limits.h>
#include <stdlib.h>

#include "../core/memory.h"
#include "tensor_module.h"

/* Number of elements in a shape tuple, or -1 with a Python error set. */
static long shape_numel(PyObject *shape)
{
    long size = 1;
    Py_ssize_t ndim = PyTuple_GET_SIZE(shape);
    for (Py_ssize_t i = 0; i < ndim; i++) {
        long dim = PyLong_AsLong(PyTuple_GET_ITEM(shape, i));
        if (dim == -1 && PyErr_Occurred()) return -1;
        if (dim < 0) {
            PyErr_SetString(PyExc_ValueError, "Dimensions cannot be negative");
            return -1;
        }
        if (dim != 0 && size > INT_MAX / dim) {
            PyErr_SetString(PyExc_ValueError, "Tensor too large (more than INT_MAX elements)");
            return -1;
        }
        size *= dim;
    }
    return size;
}

typedef void (*fill_func)(float *out, long n, float a, float b);

static void fill_uniform(float *out, long n, float low, float high)
{
    float range = high - low;
    for (long i = 0; i < n; i++) out[i] = low + ((float)rand() / (float)RAND_MAX) * range;
}

static void fill_constant(float *out, long n, float value, float unused)
{
    (void)unused;
    for (long i = 0; i < n; i++) out[i] = value;
}

/* A new tensor of `shape`, filled on the host by `fill` and uploaded. */
static PyObject *filled_tensor(PyObject *shape, fill_func fill, float a, float b)
{
    long size = shape_numel(shape);
    if (size < 0) return NULL;

    size_t bytes = (size_t)size * sizeof(float);
    float *host = (float *)malloc(bytes);
    if (!host && bytes) return PyErr_NoMemory();

    _Tensor *result = alloc_result((int)size);
    if (!result) {
        free(host);
        return NULL;
    }

    fill(host, size, a, b);
    copy_to_device(result->d_ptr, host, bytes);
    free(host);
    return (PyObject *)result;
}

static PyObject *_uniform(PyObject *module, PyObject *args)
{
    float low, high;
    PyObject *shape;
    if (!PyArg_ParseTuple(args, "ffO!", &low, &high, &PyTuple_Type, &shape)) return NULL;
    return filled_tensor(shape, fill_uniform, low, high);
}

static PyObject *_constant(PyObject *module, PyObject *args)
{
    float value;
    PyObject *shape;
    if (!PyArg_ParseTuple(args, "fO!", &value, &PyTuple_Type, &shape)) return NULL;
    return filled_tensor(shape, fill_constant, value, 0.0f);
}

static PyObject *_seed(PyObject *module, PyObject *args)
{
    int seed;
    if (!PyArg_ParseTuple(args, "i", &seed)) return NULL;
    srand(seed);
    Py_RETURN_NONE;
}

PyMethodDef random_methods[] = {
    {"_uniform", (PyCFunction)_uniform, METH_VARARGS, "Sample from U(low, high)"},
    {"_constant", (PyCFunction)_constant, METH_VARARGS, "Fill with a constant"},
    {"_seed", (PyCFunction)_seed, METH_VARARGS, "Seed the host random generator"},
    {NULL}};
