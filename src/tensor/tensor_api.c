#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>

#include "../core/memory.h"
#include "../core/tensor_types.h"
#include "operators.h"
#include "tensor_helpers.h"

#define OP_METHOD(NAME) {#NAME, (PyCFunction)NAME, METH_VARARGS, "Element-wise " #NAME " operation"}

#define TENSOR_TENSOR_OP_WRAPPER(NAME, FUNC)                \
    static PyObject *NAME(PyObject *module, PyObject *args) \
    {                                                       \
        return impl_tensor_binary_op(module, args, FUNC);   \
    }

#define TENSOR_SCALAR_OP_WRAPPER(NAME, FUNC)                \
    static PyObject *NAME(PyObject *module, PyObject *args) \
    {                                                       \
        return impl_float_binary_op(module, args, FUNC);    \
    }

#define TENSOR_OP_WRAPPER(NAME, FUNC)                       \
    static PyObject *NAME(PyObject *module, PyObject *args) \
    {                                                       \
        return _impl_tensor_unary_op(module, args, FUNC);   \
    }

#define TENSOR_TENSOR_BROADCASTED_OP_WRAPPER(NAME, FUNC)              \
    static PyObject *NAME(PyObject *module, PyObject *args)           \
    {                                                                 \
        return impl_tensor_broadcasted_binary_op(module, args, FUNC); \
    }

typedef void (*binary_tensor_op_func)(const float *, const float *, float *, int);
typedef void (*binary_scalar_op_func)(const float *, const float, float *, int);
typedef void (*unary_tensor_op_func)(const float *, float *, int);
typedef void (*binary_tensor_broadcasted_op_func)(const float *, const float *, float *,
                                                  const int *, const int *, const int *, int);

static PyTypeObject _TensorType;

// dealloc, inits, copy
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

// operator wrappers
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

static PyObject *impl_float_binary_op(PyObject *module, PyObject *args, binary_scalar_op_func op)
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

    int ndim_chk1, ndim_chk2;
    int *c_shape, *c_strides_a, *c_strides_b, ndim = 0;
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

    _Tensor *result = (_Tensor *)_TensorType.tp_alloc(&_TensorType, 0);
    if (!result) goto cleanup;

    result->size = total_elements;
    result->d_ptr = alloc_memory(result->size * sizeof(float));

    if (!result->d_ptr) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate device memory");
        goto cleanup;
    }

    op(a->d_ptr, b->d_ptr, result->d_ptr, c_shape, c_strides_a, c_strides_b, ndim);

    free(c_shape);
    free(c_strides_a);
    free(c_strides_b);

    return (PyObject *)result;

cleanup:
    if (c_shape) free(c_shape);
    if (c_strides_a) free(c_strides_a);
    if (c_strides_b) free(c_strides_b);
    return NULL;
}

// unary operators
static PyObject *_impl_tensor_unary_op(PyObject *module, PyObject *args, unary_tensor_op_func op)
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

    op(a->d_ptr, result->d_ptr, a->size);
    return (PyObject *)result;
}

// arithmetic operations
// add
TENSOR_TENSOR_OP_WRAPPER(_add_tensor, add_tensor)
TENSOR_SCALAR_OP_WRAPPER(_add_scalar, add_scalar)
TENSOR_TENSOR_BROADCASTED_OP_WRAPPER(_add_broadcasted, add_broadcasted)

// subtract
TENSOR_TENSOR_OP_WRAPPER(_sub_tensor, sub_tensor)
TENSOR_SCALAR_OP_WRAPPER(_sub_scalar, sub_scalar)
TENSOR_SCALAR_OP_WRAPPER(_rsub_scalar, rsub_scalar)
TENSOR_TENSOR_BROADCASTED_OP_WRAPPER(_sub_broadcasted, sub_broadcasted)

// multiply
TENSOR_TENSOR_OP_WRAPPER(_mul_tensor, mul_tensor)
TENSOR_SCALAR_OP_WRAPPER(_mul_scalar, mul_scalar)
TENSOR_TENSOR_BROADCASTED_OP_WRAPPER(_mul_broadcasted, mul_broadcasted)

// true division
TENSOR_TENSOR_OP_WRAPPER(_truediv_tensor, truediv_tensor)
TENSOR_SCALAR_OP_WRAPPER(_truediv_scalar, truediv_scalar)
TENSOR_SCALAR_OP_WRAPPER(_rtruediv_scalar, rtruediv_scalar)
TENSOR_TENSOR_BROADCASTED_OP_WRAPPER(_truediv_broadcasted, truediv_broadcasted)

// power
TENSOR_TENSOR_OP_WRAPPER(_pow_tensor, pow_tensor)
TENSOR_SCALAR_OP_WRAPPER(_pow_scalar, pow_scalar)
TENSOR_SCALAR_OP_WRAPPER(_rpow_scalar, rpow_scalar)

// comparison operations
// max
TENSOR_TENSOR_OP_WRAPPER(_max_tensor, max_tensor)
TENSOR_SCALAR_OP_WRAPPER(_max_scalar, max_scalar)

// min
TENSOR_TENSOR_OP_WRAPPER(_min_tensor, min_tensor)
TENSOR_SCALAR_OP_WRAPPER(_min_scalar, min_scalar)

// greater than
TENSOR_TENSOR_OP_WRAPPER(_gt_tensor, gt_tensor)
TENSOR_SCALAR_OP_WRAPPER(_gt_scalar, gt_scalar)

// less than
TENSOR_TENSOR_OP_WRAPPER(_lt_tensor, lt_tensor)
TENSOR_SCALAR_OP_WRAPPER(_lt_scalar, lt_scalar)

// unary tensor operations
// functions
TENSOR_OP_WRAPPER(_negate, negate_tensor)
TENSOR_OP_WRAPPER(_exponent, exponent_tensor)
TENSOR_OP_WRAPPER(_sqrt, sqrt_tensor)

// trigonometric
TENSOR_OP_WRAPPER(_sin, sin_tensor)
TENSOR_OP_WRAPPER(_cos, cos_tensor)
TENSOR_OP_WRAPPER(_tan, tan_tensor)
TENSOR_OP_WRAPPER(_ctg, ctg_tensor)

// trigonometric hyperbolic
TENSOR_OP_WRAPPER(_sinh, sinh_tensor)
TENSOR_OP_WRAPPER(_cosh, cosh_tensor)
TENSOR_OP_WRAPPER(_tanh, tanh_tensor)
TENSOR_OP_WRAPPER(_ctgh, ctgh_tensor)

// misc operators
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

    _Tensor *result = (_Tensor *)_TensorType.tp_alloc(&_TensorType, 0);
    if (!result) return NULL;

    result->size = n * m;
    result->d_ptr = alloc_memory(result->size * sizeof(float));

    if (!result->d_ptr) {
        Py_DECREF(result);
        return PyErr_NoMemory();
    }

    matmul(a->d_ptr, b->d_ptr, result->d_ptr, n, m, k);

    return (PyObject *)result;
}

static PyObject *_transpose(PyObject *module, PyObject *args)
{
    _Tensor *a;
    int n, m;
    if (!PyArg_ParseTuple(args, "O!ii", &_TensorType, &a, &n, &m)) return NULL;

    _Tensor *result = (_Tensor *)_TensorType.tp_alloc(&_TensorType, 0);
    if (!result) return NULL;

    result->size = a->size;
    result->d_ptr = alloc_memory(result->size * sizeof(float));

    if (!result->d_ptr) {
        Py_DECREF(result);
        return PyErr_NoMemory();
    }

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
    tuple_to_array(a_shape_obj, &a_shape, &a_dim);

    int outer_stride = sum_axis_product(a_shape, 0, axis);
    int axis_dim = a_shape[axis];
    int inner_stride = sum_axis_product(a_shape, axis + 1, a_dim);

    free(a_shape);

    _Tensor *result = (_Tensor *)_TensorType.tp_alloc(&_TensorType, 0);
    if (!result) return NULL;

    result->size = outer_stride * inner_stride;
    result->d_ptr = alloc_memory(result->size * sizeof(float));

    if (!result->d_ptr) {
        Py_DECREF(result);
        return PyErr_NoMemory();
    }

    sum_axis(a->d_ptr, result->d_ptr, outer_stride, inner_stride, axis_dim);
    return (PyObject *)result;
}

// instance methods for Tensor object
static PyMethodDef _Tensor_instance_methods[] = {
    {"_copy_from_list", (PyCFunction)_Tensor_copy_from_list, METH_VARARGS, "Load data from list"},
    {"_to_list", (PyCFunction)_Tensor_to_list, METH_VARARGS, "Export data to list"},
    {"_to_np", (PyCFunction)_Tensor_to_np, METH_VARARGS, "Copy to np array"},
    {NULL}};

// instance members for Tensor object
static PyMemberDef _Tensor_members[] = {
    {"size", T_INT, offsetof(_Tensor, size), READONLY, "Size of the tensor"}, {NULL}};

// module functions
static PyMethodDef module_methods[] = {
    // arithmetic operations
    // add
    OP_METHOD(_add_tensor),
    OP_METHOD(_add_scalar),
    OP_METHOD(_add_broadcasted),
    // subtract
    OP_METHOD(_sub_tensor),
    OP_METHOD(_sub_scalar),
    OP_METHOD(_rsub_scalar),
    OP_METHOD(_sub_broadcasted),
    // multiply
    OP_METHOD(_mul_tensor),
    OP_METHOD(_mul_scalar),
    OP_METHOD(_mul_broadcasted),
    // true division
    OP_METHOD(_truediv_tensor),
    OP_METHOD(_truediv_scalar),
    OP_METHOD(_rtruediv_scalar),
    OP_METHOD(_truediv_broadcasted),
    // power
    OP_METHOD(_pow_tensor),
    OP_METHOD(_pow_scalar),
    OP_METHOD(_rpow_scalar),

    // comparison operators
    // max
    OP_METHOD(_max_tensor),
    OP_METHOD(_max_scalar),
    // min
    OP_METHOD(_min_tensor),
    OP_METHOD(_min_scalar),
    // greater than
    OP_METHOD(_gt_tensor),
    OP_METHOD(_gt_scalar),
    // less than
    OP_METHOD(_lt_tensor),
    OP_METHOD(_lt_scalar),

    // unary operators
    // arithmetic
    OP_METHOD(_negate),
    OP_METHOD(_exponent),
    OP_METHOD(_sqrt),
    // trigonometric
    OP_METHOD(_sin),
    OP_METHOD(_cos),
    OP_METHOD(_tan),
    OP_METHOD(_ctg),
    // trigonometric hyperbolic
    OP_METHOD(_sinh),
    OP_METHOD(_cosh),
    OP_METHOD(_tanh),
    OP_METHOD(_ctgh),

    // misc
    OP_METHOD(_matmul),
    OP_METHOD(_transpose),
    OP_METHOD(_from_numpy),

    // summations
    OP_METHOD(_sum),
    OP_METHOD(_sum_axis),

    // end
    {NULL}};

// Tensor type
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

// _tensor module
static PyModuleDef tensor_module = {
    PyModuleDef_HEAD_INIT, .m_name = "ember._core._tensor", .m_doc = "Ember Tensor backend",
    .m_size = -1,          .m_methods = module_methods,
};

// initialize module
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
