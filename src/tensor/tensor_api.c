#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>

#include "../core/memory.h"
#include "../core/tensor_types.h"
#include "operators.h"
#include "tensor_helpers.h"

#define OP_METHOD(NAME) {#NAME, (PyCFunction)NAME, METH_VARARGS, "Element-wise " #NAME " operation"}

typedef void (*binary_tensor_op_func)(const float *, const float *, float *, int);
typedef void (*binary_scalar_op_func)(const float *, float, float *, int);
typedef void (*unary_tensor_op_func)(const float *, float *, int);
typedef void (*binary_tensor_broadcasted_op_func)(const float *, const float *, float *,
                                                  const int *, const int *, const int *, int);

static PyTypeObject _TensorType;

/* ---- allocation helper: a result _Tensor with `size` device floats ---- */
static _Tensor *alloc_result(int size)
{
    _Tensor *result = (_Tensor *)_TensorType.tp_alloc(&_TensorType, 0);
    if (!result) return NULL;

    result->size = size;
    result->d_ptr = alloc_memory((size_t)size * sizeof(float));
    if (!result->d_ptr) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate device memory");
        return NULL;
    }
    return result;
}

/* ---- dunder methods ---- */
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
    self->d_ptr = alloc_memory((size_t)size * sizeof(float));

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

    float *temp_host = (float *)malloc((size_t)self->size * sizeof(float));
    if (!temp_host) return PyErr_NoMemory();

    for (int i = 0; i < self->size; i++) {
        PyObject *item = PyList_GetItem(py_list, i);
        temp_host[i] = (float)PyFloat_AsDouble(item);
    }

    copy_to_device(self->d_ptr, temp_host, (size_t)self->size * sizeof(float));
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
    if (!c_dims) return PyErr_NoMemory();
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

    float *temp_host = (float *)malloc((size_t)self->size * sizeof(float));
    if (!temp_host) {
        free(c_dims);
        return PyErr_NoMemory();
    }

    sync_device();  // make sure the device is synchronized
    copy_from_device(temp_host, self->d_ptr, (size_t)self->size * sizeof(float));

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

    PyObject *arr = PyArray_SimpleNew(1, dims, NPY_FLOAT);
    if (arr == NULL) {
        return NULL;
    }

    void *dst = PyArray_DATA((PyArrayObject *)arr);

    sync_device();  // make sure the device is synchronized
    copy_from_device(dst, self->d_ptr, (size_t)self->size * sizeof(float));

    return arr;
}

static PyObject *_from_numpy(PyObject *module, PyObject *args)
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

    int num_elements = (int)(view.len / view.itemsize);
    _Tensor *result = alloc_result(num_elements);
    if (!result) {
        PyBuffer_Release(&view);
        return NULL;
    }

    copy_to_device(result->d_ptr, view.buf, (size_t)result->size * sizeof(float));
    PyBuffer_Release(&view);

    return (PyObject *)result;
}

/* ---- generic operator dispatchers ---- */
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

    _Tensor *result = alloc_result(a->size);
    if (!result) return NULL;

    op(a->d_ptr, b->d_ptr, result->d_ptr, a->size);
    return (PyObject *)result;
}

static PyObject *impl_float_binary_op(PyObject *module, PyObject *args, binary_scalar_op_func op)
{
    _Tensor *a;
    float b;

    if (!PyArg_ParseTuple(args, "O!f", &_TensorType, &a, &b)) {
        return NULL;
    }

    _Tensor *result = alloc_result(a->size);
    if (!result) return NULL;

    op(a->d_ptr, b, result->d_ptr, a->size);
    return (PyObject *)result;
}

static PyObject *impl_tensor_broadcasted_binary_op(PyObject *module, PyObject *args,
                                                   binary_tensor_broadcasted_op_func op)
{
    _Tensor *a, *b;
    PyObject *shape_tuple, *strides_a_tuple, *strides_b_tuple;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &_TensorType, &a, &_TensorType, &b, &PyTuple_Type,
                          &shape_tuple, &PyTuple_Type, &strides_a_tuple, &PyTuple_Type,
                          &strides_b_tuple)) {
        return NULL;
    }

    int *c_shape = NULL, *c_strides_a = NULL, *c_strides_b = NULL;
    int ndim = 0, ndim_chk1 = 0, ndim_chk2 = 0;
    if (tuple_to_array(shape_tuple, &c_shape, &ndim) < 0) goto cleanup;
    if (tuple_to_array(strides_a_tuple, &c_strides_a, &ndim_chk1) < 0) goto cleanup;
    if (tuple_to_array(strides_b_tuple, &c_strides_b, &ndim_chk2) < 0) goto cleanup;

    if (ndim != ndim_chk1 || ndim != ndim_chk2) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch in broadcast arguments");
        goto cleanup;
    }

    int total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        total_elements *= c_shape[i];
    }

    _Tensor *result = alloc_result(total_elements);
    if (!result) goto cleanup;

    op(a->d_ptr, b->d_ptr, result->d_ptr, c_shape, c_strides_a, c_strides_b, ndim);

    free(c_shape);
    free(c_strides_a);
    free(c_strides_b);
    return (PyObject *)result;

cleanup:
    free(c_shape);
    free(c_strides_a);
    free(c_strides_b);
    return NULL;
}

static PyObject *impl_tensor_unary_op(PyObject *module, PyObject *args, unary_tensor_op_func op)
{
    _Tensor *a;
    if (!PyArg_ParseTuple(args, "O!", &_TensorType, &a)) return NULL;

    _Tensor *result = alloc_result(a->size);
    if (!result) return NULL;

    op(a->d_ptr, result->d_ptr, a->size);
    return (PyObject *)result;
}

/* ---- Python wrappers, generated from operators.def ---- */
#define EMBER_BINARY_OP(name, expr)                                \
    static PyObject *_##name##_tensor(PyObject *m, PyObject *args) \
    {                                                              \
        return impl_tensor_binary_op(m, args, name##_tensor);      \
    }
#define EMBER_SCALAR_OP(name, expr)                                \
    static PyObject *_##name##_scalar(PyObject *m, PyObject *args) \
    {                                                              \
        return impl_float_binary_op(m, args, name##_scalar);       \
    }
#define EMBER_BROADCAST_OP(name, expr)                                         \
    static PyObject *_##name##_broadcasted(PyObject *m, PyObject *args)        \
    {                                                                          \
        return impl_tensor_broadcasted_binary_op(m, args, name##_broadcasted); \
    }
#define EMBER_UNARY_OP(name, expr)                           \
    static PyObject *_##name(PyObject *m, PyObject *args)    \
    {                                                        \
        return impl_tensor_unary_op(m, args, name##_tensor); \
    }
#include "operators.def"

/* ---- non-element-wise operators ---- */
static PyObject *_matmul(PyObject *module, PyObject *args)
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

    _Tensor *result = alloc_result(n * m);
    if (!result) return NULL;

    matmul(a->d_ptr, b->d_ptr, result->d_ptr, n, m, k);
    return (PyObject *)result;
}

static PyObject *_transpose(PyObject *module, PyObject *args)
{
    _Tensor *a;
    int n, m;
    if (!PyArg_ParseTuple(args, "O!ii", &_TensorType, &a, &n, &m)) return NULL;

    _Tensor *result = alloc_result(a->size);
    if (!result) return NULL;

    transpose(a->d_ptr, result->d_ptr, n, m);
    return (PyObject *)result;
}

static PyObject *_sum(PyObject *module, PyObject *args)
{
    _Tensor *a;
    if (!PyArg_ParseTuple(args, "O!", &_TensorType, &a)) return NULL;

    float result = sum(a->d_ptr, a->size);
    return PyFloat_FromDouble((double)result);
}

static PyObject *_sum_axis(PyObject *module, PyObject *args)
{
    _Tensor *a;
    PyObject *a_shape_obj;
    int axis;

    if (!PyArg_ParseTuple(args, "O!O!i", &_TensorType, &a, &PyTuple_Type, &a_shape_obj, &axis))
        return NULL;

    int a_dim, *a_shape;
    if (tuple_to_array(a_shape_obj, &a_shape, &a_dim) < 0) return NULL;

    int outer_stride = sum_axis_product(a_shape, 0, axis);
    int axis_dim = a_shape[axis];
    int inner_stride = sum_axis_product(a_shape, axis + 1, a_dim);

    free(a_shape);

    _Tensor *result = alloc_result(outer_stride * inner_stride);
    if (!result) return NULL;

    sum_axis(a->d_ptr, result->d_ptr, outer_stride, inner_stride, axis_dim);
    return (PyObject *)result;
}

/* ---- type & module definitions ---- */
static PyMethodDef _Tensor_instance_methods[] = {
    {"_copy_from_list", (PyCFunction)_Tensor_copy_from_list, METH_VARARGS, "Load data from list"},
    {"_to_list", (PyCFunction)_Tensor_to_list, METH_VARARGS, "Export data to list"},
    {"_to_np", (PyCFunction)_Tensor_to_np, METH_VARARGS, "Copy to np array"},
    {NULL}};

static PyMemberDef _Tensor_members[] = {
    {"size", T_INT, offsetof(_Tensor, size), READONLY, "Size of the tensor"}, {NULL}};

static PyMethodDef module_methods[] = {
/* Element-wise operator table, generated from operators.def. */
#define EMBER_BINARY_OP(name, expr) OP_METHOD(_##name##_tensor),
#define EMBER_SCALAR_OP(name, expr) OP_METHOD(_##name##_scalar),
#define EMBER_BROADCAST_OP(name, expr) OP_METHOD(_##name##_broadcasted),
#define EMBER_UNARY_OP(name, expr) OP_METHOD(_##name),
#include "operators.def"

    /* non-element-wise operators */
    OP_METHOD(_matmul),
    OP_METHOD(_transpose),
    OP_METHOD(_from_numpy),
    OP_METHOD(_sum),
    OP_METHOD(_sum_axis),

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
