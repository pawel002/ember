#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>

#include "../core/memory.h"
#include "ops.h"
#include "tensor_helpers.h"

typedef void (*binary_tensor_op_func)(const float *, const float *, float *, int);
typedef void (*binary_float_op_func)(const float *, const float, float *, int);

static PyTypeObject _TensorType;

typedef struct {
    PyObject_HEAD void *d_ptr;
    int size;
} _Tensor;

// --- Internal Object Methods (Dealloc/Init/IO) ---

static void _Tensor_dealloc(_Tensor *self)
{
    if (self->d_ptr) free_memory(self->d_ptr);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int _Tensor_init(_Tensor *self, PyObject *args, PyObject *kwds)
{
    int size = 0;
    if (!PyArg_ParseTuple(args, "i", &size)) {
        return -1;
    }

    self->size = size;
    self->d_ptr = alloc_memory(size * sizeof(float));

    if (self->d_ptr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate device memory");
        return -1;
    }
    return 0;
}

static PyObject *_Tensor_copy_from_list(_Tensor *self, PyObject *args)
{
    PyObject *py_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &py_list)) return NULL;

    float *temp_host = (float *)malloc(self->size * sizeof(float));

    for (int i = 0; i < self->size; i++) {
        PyObject *item = PyList_GetItem(py_list, i);
        temp_host[i] = (float)PyFloat_AsDouble(item);
    }

    copy_to_device(self->d_ptr, temp_host, self->size * sizeof(float));
    free(temp_host);

    Py_RETURN_NONE;
}

static PyObject *_Tensor_to_list(_Tensor *self, PyObject *args)
{
    PyObject *shape_tuple;

    if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &shape_tuple)) {
        return NULL;
    }

    Py_ssize_t ndim = PyTuple_Size(shape_tuple);
    long *c_dims = (long *)malloc(ndim * sizeof(long));
    long total_elements = 1;

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject *item = PyTuple_GetItem(shape_tuple, i);
        long dim = PyLong_AsLong(item);

        if (dim < 0) {
            free(c_dims);
            PyErr_SetString(PyExc_ValueError, "Dimensions cannot be negative");
            return NULL;
        }

        c_dims[i] = dim;
        total_elements *= dim;
    }

    if (total_elements != self->size) {
        free(c_dims);
        PyErr_Format(PyExc_ValueError,
                     "Shape mismatch: Tensor has %d elements, but requested shape requires %ld",
                     self->size, total_elements);
        return NULL;
    }

    float *temp_host = (float *)malloc(self->size * sizeof(float));
    if (!temp_host) {
        free(c_dims);
        return PyErr_NoMemory();
    }

    sync_device();  // make sure the device is synchronized
    copy_from_device(temp_host, self->d_ptr, self->size * sizeof(float));

    int offset = 0;
    PyObject *result = build_nested_list(temp_host, c_dims, (int)ndim, 0, &offset);

    free(temp_host);
    free(c_dims);

    return result;
}

static PyObject *_Tensor_to_np(_Tensor *self, PyObject *args)
{
    import_array();
    npy_intp dims[1];
    dims[0] = self->size;

    int type_num = NPY_FLOAT;
    PyObject *arr = PyArray_SimpleNew(1, dims, type_num);

    if (arr == NULL) {
        return NULL;
    }

    void *dst = PyArray_DATA((PyArrayObject *)arr);

    sync_device();  // make sure the device is synchronized
    copy_from_device(dst, self->d_ptr, (size_t)self->size * sizeof(float));

    return arr;
}

// --- Fancy innits (ie. from numpy)

static PyObject *_tensor_from_numpy(PyObject *module, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }

    Py_buffer view;
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (PyObject_GetBuffer(obj, &view, flags) < 0) {
        PyErr_SetString(PyExc_TypeError,
                        "Object does not support the buffer protocol (is it a numpy array?)");
        return NULL;
    }

    if (view.format != NULL && strcmp(view.format, "f") != 0) {
        PyBuffer_Release(&view);
        PyErr_Format(PyExc_TypeError,
                     "Expected float32 ('f'), got '%s'. Use array.astype('float32') in Python.",
                     view.format);
        return NULL;
    }

    int num_elements = view.len / view.itemsize;
    _Tensor *result = (_Tensor *)_TensorType.tp_alloc(&_TensorType, 0);
    if (!result) {
        PyBuffer_Release(&view);
        return NULL;
    }

    result->size = (int)num_elements;
    result->d_ptr = alloc_memory(result->size * sizeof(float));

    if (!result->d_ptr) {
        Py_DECREF(result);
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }

    copy_to_device(result->d_ptr, view.buf, result->size * sizeof(float));
    PyBuffer_Release(&view);

    return (PyObject *)result;
}

// --- Standalone Module Functions ---

static PyObject *impl_tensor_binary_op(PyObject *module, PyObject *args, binary_tensor_op_func op)
{
    _Tensor *a, *b;
    if (!PyArg_ParseTuple(args, "O!O!", &_TensorType, &a, &_TensorType, &b)) {
        return NULL;
    }

    if (a->size != b->size) {
        PyErr_Format(PyExc_ValueError, "Size mismatch: %d vs %d", a->size, b->size);
        return NULL;
    }

    _Tensor *result = (_Tensor *)_TensorType.tp_alloc(&_TensorType, 0);
    if (!result) return NULL;

    result->size = a->size;
    result->d_ptr = alloc_memory(result->size * sizeof(float));

    if (!result->d_ptr) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate device memory");
        return NULL;
    }

    op(a->d_ptr, b->d_ptr, result->d_ptr, a->size);
    return (PyObject *)result;
}

static PyObject *impl_float_binary_op(PyObject *module, PyObject *args, binary_float_op_func op)
{
    _Tensor *a;
    float b;

    if (!PyArg_ParseTuple(args, "O!f", &_TensorType, &a, &b)) {
        return NULL;
    }

    _Tensor *result = (_Tensor *)_TensorType.tp_alloc(&_TensorType, 0);
    if (!result) return NULL;

    result->size = a->size;
    result->d_ptr = alloc_memory(result->size * sizeof(float));

    if (!result->d_ptr) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate device memory");
        return NULL;
    }

    op(a->d_ptr, b, result->d_ptr, a->size);
    return (PyObject *)result;
}

// Tensor operators
static PyObject *_add_tensor(PyObject *module, PyObject *args)
{
    return impl_tensor_binary_op(module, args, add_tensor);
}

static PyObject *_sub_tensor(PyObject *module, PyObject *args)
{
    return impl_tensor_binary_op(module, args, sub_tensor);
}

static PyObject *_mul_tensor(PyObject *module, PyObject *args)
{
    return impl_tensor_binary_op(module, args, mul_tensor);
}

static PyObject *_max_tensor(PyObject *module, PyObject *args)
{
    return impl_tensor_binary_op(module, args, max_tensor);
}

static PyObject *_min_tensor(PyObject *module, PyObject *args)
{
    return impl_tensor_binary_op(module, args, min_tensor);
}

static PyObject *_gt_tensor(PyObject *module, PyObject *args)
{
    return impl_tensor_binary_op(module, args, gt_tensor);
}

// Scalar operators
static PyObject *_add_scalar(PyObject *module, PyObject *args)
{
    return impl_float_binary_op(module, args, add_scalar);
}

static PyObject *_sub_scalar(PyObject *module, PyObject *args)
{
    return impl_float_binary_op(module, args, sub_scalar);
}

static PyObject *_rsub_scalar(PyObject *module, PyObject *args)
{
    return impl_float_binary_op(module, args, rsub_scalar);
}

static PyObject *_mul_scalar(PyObject *module, PyObject *args)
{
    return impl_float_binary_op(module, args, mul_scalar);
}

static PyObject *_max_scalar(PyObject *module, PyObject *args)
{
    return impl_float_binary_op(module, args, max_scalar);
}

static PyObject *_min_scalar(PyObject *module, PyObject *args)
{
    return impl_float_binary_op(module, args, min_scalar);
}

static PyObject *_gt_scalar(PyObject *module, PyObject *args)
{
    return impl_float_binary_op(module, args, gt_scalar);
}

// Misc operators
static PyObject *_simple_matmul(PyObject *module, PyObject *args)
{
    _Tensor *a, *b;
    int n, m, k;

    if (!PyArg_ParseTuple(args, "O!O!iii", &_TensorType, &a, &_TensorType, &b, &n, &m, &k)) {
        return NULL;
    }

    if (a->size != n * k) {
        PyErr_Format(PyExc_ValueError, "Shape mismatch A: %d != %d x %d", a->size, n, k);
        return NULL;
    }
    if (b->size != k * m) {
        PyErr_Format(PyExc_ValueError, "Shape mismatch B: %d != %d x %d", b->size, k, m);
        return NULL;
    }

    _Tensor *result = (_Tensor *)_TensorType.tp_alloc(&_TensorType, 0);
    if (!result) return NULL;

    result->size = n * m;
    result->d_ptr = alloc_memory(result->size * sizeof(float));

    if (!result->d_ptr) {
        Py_DECREF(result);
        return PyErr_NoMemory();
    }

    simple_matmul(a->d_ptr, b->d_ptr, result->d_ptr, n, m, k);

    return (PyObject *)result;
}

static PyObject *_negate(PyObject *module, PyObject *args)
{
    _Tensor *a;
    if (!PyArg_ParseTuple(args, "O!", &_TensorType, &a)) return NULL;

    _Tensor *result = (_Tensor *)_TensorType.tp_alloc(&_TensorType, 0);
    if (!result) return NULL;

    result->size = a->size;
    result->d_ptr = alloc_memory(result->size * sizeof(float));

    if (!result->d_ptr) {
        Py_DECREF(result);
        return PyErr_NoMemory();
    }

    negate(a->d_ptr, result->d_ptr, a->size);

    return (PyObject *)result;
}

// --- Method Tables ---
// Methods attached to the _Tensor OBJECT (Instance methods)
static PyMethodDef _Tensor_instance_methods[] = {
    {"_copy_from_list", (PyCFunction)_Tensor_copy_from_list, METH_VARARGS, "Load data from list"},
    {"_to_list", (PyCFunction)_Tensor_to_list, METH_VARARGS, "Export data to list"},
    {"_to_np", (PyCFunction)_Tensor_to_np, METH_VARARGS, "Copy to np array"},
    {NULL}};

static PyMemberDef _Tensor_members[] = {
    {"size", T_INT, offsetof(_Tensor, size), READONLY, "Size of the tensor"}, {NULL}};

// Methods attached to the MODULE (Standalone functions)
static PyMethodDef module_methods[] = {
    {"_add_tensor", (PyCFunction)_add_tensor, METH_VARARGS, "T + T"},
    {"_sub_tensor", (PyCFunction)_sub_tensor, METH_VARARGS, "T - T"},
    {"_mul_tensor", (PyCFunction)_mul_tensor, METH_VARARGS, "T * T"},
    {"_max_tensor", (PyCFunction)_max_tensor, METH_VARARGS, "max(T, T)"},
    {"_min_tensor", (PyCFunction)_min_tensor, METH_VARARGS, "min(T, T)"},
    {"_gt_tensor", (PyCFunction)_gt_tensor, METH_VARARGS, "T > T"},
    {"_add_scalar", (PyCFunction)_add_scalar, METH_VARARGS, "T + float"},
    {"_sub_scalar", (PyCFunction)_sub_scalar, METH_VARARGS, "T - float"},
    {"_rsub_scalar", (PyCFunction)_rsub_scalar, METH_VARARGS, "float - T"},
    {"_mul_scalar", (PyCFunction)_mul_scalar, METH_VARARGS, "T * float"},
    {"_max_scalar", (PyCFunction)_max_scalar, METH_VARARGS, "max(T, float)"},
    {"_min_scalar", (PyCFunction)_min_scalar, METH_VARARGS, "min(T, float)"},
    {"_gt_scalar", (PyCFunction)_gt_scalar, METH_VARARGS, "T > float"},
    {"_negate", (PyCFunction)_negate, METH_VARARGS, "-T"},
    {"_matmul", (PyCFunction)_simple_matmul, METH_VARARGS, "T @ T"},
    {"_from_numpy", (PyCFunction)_tensor_from_numpy, METH_VARARGS, "T from np"},
    {NULL}};

static PyTypeObject _TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "ember._core._tensor._Tensor",
    .tp_basicsize = sizeof(_Tensor),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Internal Tensor Object",
    .tp_methods = _Tensor_instance_methods,
    .tp_members = _Tensor_members,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)_Tensor_init,
    .tp_dealloc = (destructor)_Tensor_dealloc,
};

static PyModuleDef tensor_module = {
    PyModuleDef_HEAD_INIT, .m_name = "ember._core._tensor", .m_doc = "Ember Tensor backend",
    .m_size = -1,          .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit__tensor(void)
{
    if (PyType_Ready(&_TensorType) < 0) return NULL;

    PyObject *m = PyModule_Create(&tensor_module);
    if (!m) return NULL;

    Py_INCREF(&_TensorType);
    if (PyModule_AddObject(m, "_Tensor", (PyObject *)&_TensorType) < 0) {
        Py_DECREF(&_TensorType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
