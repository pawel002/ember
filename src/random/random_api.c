#include <Python.h>
#include <stdlib.h>
#include <time.h>

#include "../core/memory.h"
#include "../core/tensor_types.h"
#include "distributions.h"

// get _Tensor type
static PyTypeObject *_get_tensor_type(void)
{
    PyObject *module = PyImport_ImportModule("ember._core._tensor");
    if (!module) return NULL;

    PyObject *cls = PyObject_GetAttrString(module, "_Tensor");
    Py_DECREF(module);

    return (PyTypeObject *)cls;
}

// allocate tensor with given shape
static PyObject *_alloc_tensor_with_shape(PyObject *shape_tuple, long *out_size)
{
    long size = 1;
    Py_ssize_t ndim = PyTuple_Size(shape_tuple);

    for (Py_ssize_t i = 0; i < ndim; i++) {
        long dim = PyLong_AsLong(PyTuple_GetItem(shape_tuple, i));
        if (dim < 0) {
            PyErr_SetString(PyExc_ValueError, "Dimensions cannot be negative");
            return NULL;
        }
        size *= dim;
    }

    PyTypeObject *TensorType = _get_tensor_type();
    if (!TensorType) return NULL;

    PyObject *tensor_obj = PyObject_CallFunction((PyObject *)TensorType, "i", (int)size);
    Py_DECREF(TensorType);
    if (!tensor_obj) return NULL;

    _Tensor *tensor = (_Tensor *)tensor_obj;
    tensor->d_ptr = alloc_memory((size_t)size * sizeof(float));

    if (!tensor->d_ptr) {
        Py_DECREF(tensor_obj);
        return PyErr_NoMemory();
    }

    *out_size = size;
    return tensor_obj;
}

// distributions
static PyObject *_uniform(PyObject *self, PyObject *args)
{
    float low, high;
    PyObject *shape_tuple;

    if (!PyArg_ParseTuple(args, "ffO!", &low, &high, &PyTuple_Type, &shape_tuple)) {
        return NULL;
    }

    long size;
    PyObject *tensor_obj = _alloc_tensor_with_shape(shape_tuple, &size);
    if (!tensor_obj) return NULL;

    _Tensor *tensor = (_Tensor *)tensor_obj;
    uniform(low, high, tensor->d_ptr, size);
    return tensor_obj;
}

static PyObject *_constant(PyObject *self, PyObject *args)
{
    float value;
    PyObject *shape_tuple;

    if (!PyArg_ParseTuple(args, "fO!", &value, &PyTuple_Type, &shape_tuple)) {
        return NULL;
    }

    long size;
    PyObject *tensor_obj = _alloc_tensor_with_shape(shape_tuple, &size);
    if (!tensor_obj) return NULL;

    _Tensor *tensor = (_Tensor *)tensor_obj;
    constant(value, tensor->d_ptr, size);
    return tensor_obj;
}

static PyObject *_seed(PyObject *self, PyObject *args)
{
    int seed;
    if (!PyArg_ParseTuple(args, "i", &seed)) {
        return NULL;
    }
    srand(seed);
    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"_uniform", (PyCFunction)_uniform, METH_VARARGS, "Sample from U(a, b)"},
    {"_constant", (PyCFunction)_constant, METH_VARARGS, "Const sample"},
    {"_seed", (PyCFunction)_seed, METH_VARARGS, "Set the seed"},
    {NULL, NULL, 0, NULL}};

static PyModuleDef random_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "ember._core._em_random",
    .m_doc = "Random number generation module",
    .m_size = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit__em_random(void)
{
    PyObject *m = PyModule_Create(&random_module);
    if (!m) return NULL;

    return m;
}
