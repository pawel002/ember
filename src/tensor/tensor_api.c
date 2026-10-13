#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>

#include "../core/memory.h"
#include "operators.h"
#include "tensor_helpers.h"
#include "tensor_module.h"
#include "transformer.h"

typedef void (*binary_tensor_op_func)(const float *, const float *, float *, int);
typedef void (*binary_scalar_op_func)(const float *, float, float *, int);
typedef void (*unary_tensor_op_func)(const float *, float *, int);
typedef void (*binary_tensor_broadcasted_op_func)(const float *, const float *, float *,
                                                  const int *, const int *, const int *, int);
typedef void (*inplace_tensor_op_func)(float *, const float *, int);
typedef void (*inplace_scalar_op_func)(float *, float, int);

/* Point `t` at a fresh buffer of `size` device floats. An empty tensor holds no
 * buffer (d_ptr stays NULL) and every op skips the kernel for it. Returns 0, or
 * -1 with a Python error set. */
static int tensor_alloc(_Tensor *t, int size)
{
    if (size < 0) {
        PyErr_SetString(PyExc_ValueError, "tensor size cannot be negative");
        return -1;
    }
    t->size = size;
    t->d_ptr = NULL;
    if (size == 0) return 0;

    t->d_ptr = alloc_memory((size_t)size * sizeof(float));
    if (!t->d_ptr) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate device memory");
        return -1;
    }
    return 0;
}

/* ---- allocation helper: a result _Tensor with `size` device floats ---- */
_Tensor *alloc_result(int size)
{
    _Tensor *result = (_Tensor *)_TensorType.tp_alloc(&_TensorType, 0);
    if (!result) return NULL;

    if (tensor_alloc(result, size) < 0) {
        Py_DECREF(result);
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

    if (self->d_ptr) free_memory(self->d_ptr);  // __init__ called again
    return tensor_alloc(self, size);
}

static PyObject *_Tensor_to_np(_Tensor *self, PyObject *args)
{
    npy_intp dims[1];
    dims[0] = self->size;

    PyObject *arr = PyArray_SimpleNew(1, dims, NPY_FLOAT);
    if (arr == NULL) {
        return NULL;
    }

    void *dst = PyArray_DATA((PyArrayObject *)arr);
    copy_from_device(dst, self->d_ptr, (size_t)self->size * sizeof(float));

    return arr;
}

// Update this tensor's existing device buffer in place from a float32 numpy
// array (via the pinned async upload path). Used to feed fresh batches into a
// stable buffer, including one captured by a CUDA graph.
static PyObject *_Tensor_copy_from_numpy(_Tensor *self, PyObject *args)
{
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;

    Py_buffer view;
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (PyObject_GetBuffer(obj, &view, flags) < 0) {
        PyErr_SetString(PyExc_TypeError, "Expected a C-contiguous numpy array");
        return NULL;
    }
    if (view.format != NULL && strcmp(view.format, "f") != 0) {
        PyBuffer_Release(&view);
        PyErr_Format(PyExc_TypeError, "Expected float32 ('f'), got '%s'", view.format);
        return NULL;
    }

    Py_ssize_t n = view.len / view.itemsize;
    if (n != self->size) {
        PyBuffer_Release(&view);
        PyErr_Format(PyExc_ValueError, "Size mismatch: array has %zd elements, tensor has %d", n,
                     self->size);
        return NULL;
    }

    copy_to_device_pinned(self->d_ptr, view.buf, (size_t)self->size * sizeof(float));
    PyBuffer_Release(&view);
    Py_RETURN_NONE;
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

    int num_elements = checked_size(view.len / view.itemsize);
    _Tensor *result = num_elements < 0 ? NULL : alloc_result(num_elements);
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

    if (a->size) op(a->d_ptr, b->d_ptr, result->d_ptr, a->size);
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

    if (a->size) op(a->d_ptr, b, result->d_ptr, a->size);
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
    if (ndim > EMBER_MAX_BCAST_DIMS) {
        PyErr_Format(PyExc_ValueError, "broadcasting supports at most %d dimensions, got %d",
                     EMBER_MAX_BCAST_DIMS, ndim);
        goto cleanup;
    }

    /* The furthest element each operand is read at must lie inside it. */
    long long total = 1, last_a = 0, last_b = 0;
    for (int i = 0; i < ndim; i++) {
        total *= c_shape[i];
        if (c_shape[i] == 0) continue;
        last_a += (long long)(c_shape[i] - 1) * c_strides_a[i];
        last_b += (long long)(c_shape[i] - 1) * c_strides_b[i];
    }
    if (checked_size(total) < 0) goto cleanup;
    if (total > 0 && (last_a >= a->size || last_b >= b->size)) {
        PyErr_SetString(PyExc_ValueError, "broadcast shape/strides exceed the operand sizes");
        goto cleanup;
    }

    _Tensor *result = alloc_result((int)total);
    if (!result) goto cleanup;

    if (total) op(a->d_ptr, b->d_ptr, result->d_ptr, c_shape, c_strides_a, c_strides_b, ndim);

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

    if (a->size) op(a->d_ptr, result->d_ptr, a->size);
    return (PyObject *)result;
}

/* In-place ops mutate the first operand's buffer and allocate nothing. */
static PyObject *impl_inplace_tensor_op(PyObject *args, inplace_tensor_op_func op)
{
    _Tensor *a, *b;
    if (!PyArg_ParseTuple(args, "O!O!", &_TensorType, &a, &_TensorType, &b)) return NULL;

    if (a->size != b->size) {
        PyErr_Format(PyExc_ValueError, "Size mismatch: %d vs %d", a->size, b->size);
        return NULL;
    }

    if (a->size) op(a->d_ptr, b->d_ptr, a->size);
    Py_RETURN_NONE;
}

static PyObject *impl_inplace_scalar_op(PyObject *args, inplace_scalar_op_func op)
{
    _Tensor *a;
    float b;
    if (!PyArg_ParseTuple(args, "O!f", &_TensorType, &a, &b)) return NULL;

    if (a->size) op(a->d_ptr, b, a->size);
    Py_RETURN_NONE;
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
#define EMBER_INPLACE_OP(name, expr)                         \
    static PyObject *_##name(PyObject *m, PyObject *args)    \
    {                                                        \
        return impl_inplace_tensor_op(args, name##_inplace); \
    }
#define EMBER_INPLACE_SCALAR_OP(name, expr)                         \
    static PyObject *_##name##_scalar(PyObject *m, PyObject *args)  \
    {                                                               \
        return impl_inplace_scalar_op(args, name##_scalar_inplace); \
    }
#include "operators.def"

/* ---- non-element-wise operators ---- */
static PyObject *_matmul(PyObject *module, PyObject *args)
{
    _Tensor *a, *b;
    int n, m, k;
    if (!PyArg_ParseTuple(args, "O!O!iii", &_TensorType, &a, &_TensorType, &b, &n, &m, &k))
        return NULL;

    if (require_size(a, (long long)n * k, "A") < 0 || require_size(b, (long long)k * m, "B") < 0)
        return NULL;
    int size = checked_size((long long)n * m);
    _Tensor *result = size < 0 ? NULL : alloc_result(size);
    if (!result) return NULL;

    if (size) matmul(a->d_ptr, b->d_ptr, result->d_ptr, n, m, k);
    return (PyObject *)result;
}

static PyObject *_matmul_bias(PyObject *module, PyObject *args)
{
    _Tensor *a, *b, *bias;
    int n, m, k;
    if (!PyArg_ParseTuple(args, "O!O!O!iii", &_TensorType, &a, &_TensorType, &b, &_TensorType,
                          &bias, &n, &m, &k))
        return NULL;

    if (require_size(a, (long long)n * k, "A") < 0 || require_size(b, (long long)k * m, "B") < 0 ||
        require_size(bias, m, "bias") < 0)
        return NULL;
    int size = checked_size((long long)n * m);
    _Tensor *result = size < 0 ? NULL : alloc_result(size);
    if (!result) return NULL;

    if (size) matmul_bias(a->d_ptr, b->d_ptr, bias->d_ptr, result->d_ptr, n, m, k);
    return (PyObject *)result;
}

static PyObject *_matmul_batched(PyObject *module, PyObject *args)
{
    _Tensor *a, *b;
    int batch, n, m, k;
    if (!PyArg_ParseTuple(args, "O!O!iiii", &_TensorType, &a, &_TensorType, &b, &batch, &n, &m, &k))
        return NULL;

    if (require_size(a, (long long)batch * n * k, "A") < 0 ||
        require_size(b, (long long)batch * k * m, "B") < 0)
        return NULL;
    int size = checked_size((long long)batch * n * m);
    if (size > 0 && k == 0) {
        PyErr_SetString(PyExc_ValueError, "batched matmul needs a non-empty inner dimension");
        return NULL;
    }
    _Tensor *result = size < 0 ? NULL : alloc_result(size);
    if (!result) return NULL;

    if (size) bmm(a->d_ptr, b->d_ptr, result->d_ptr, batch, n, m, k, 0, 0, 1.0f);
    return (PyObject *)result;
}

static PyObject *_transpose(PyObject *module, PyObject *args)
{
    _Tensor *a;
    int n, m;
    if (!PyArg_ParseTuple(args, "O!ii", &_TensorType, &a, &n, &m)) return NULL;
    if (require_size(a, (long long)n * m, "input") < 0) return NULL;

    _Tensor *result = alloc_result(a->size);
    if (!result) return NULL;

    if (a->size) transpose(a->d_ptr, result->d_ptr, n, m);
    return (PyObject *)result;
}

static PyObject *_sum(PyObject *module, PyObject *args)
{
    _Tensor *a;
    if (!PyArg_ParseTuple(args, "O!", &_TensorType, &a)) return NULL;

    float result = a->size ? sum(a->d_ptr, a->size) : 0.0f;
    return PyFloat_FromDouble((double)result);
}

typedef void (*axis_reduce_func)(const float *, float *, int, int, int);

/* `empty_ok`: whether reducing over a zero-length axis has a value (0 for a
 * sum; a max has none). */
static PyObject *impl_axis_reduce(PyObject *args, axis_reduce_func op, int empty_ok)
{
    _Tensor *a;
    PyObject *shape;
    int axis;
    if (!PyArg_ParseTuple(args, "O!O!i", &_TensorType, &a, &PyTuple_Type, &shape, &axis))
        return NULL;

    int outer, axis_dim, inner;
    if (parse_axis(shape, axis, a->size, &outer, &axis_dim, &inner) < 0) return NULL;

    int size = outer * inner;  // <= a->size unless axis_dim == 0
    if (axis_dim == 0 && size > 0 && !empty_ok) {
        PyErr_SetString(PyExc_ValueError, "cannot reduce over an empty axis");
        return NULL;
    }
    if (checked_size((long long)outer * inner) < 0) return NULL;
    _Tensor *result = alloc_result(size);
    if (!result) return NULL;

    if (size) op(a->d_ptr, result->d_ptr, outer, inner, axis_dim);
    return (PyObject *)result;
}

static PyObject *_sum_axis(PyObject *module, PyObject *args)
{
    return impl_axis_reduce(args, sum_axis, 1);
}

static PyObject *_max_axis(PyObject *module, PyObject *args)
{
    return impl_axis_reduce(args, max_axis, 0);
}

/* GELU backward (grad, x, y) -> out; the other activation backprops are binary
 * ops generated from operators.def. */
static PyObject *_gelu_bwd(PyObject *module, PyObject *args)
{
    _Tensor *grad, *x;
    if (!PyArg_ParseTuple(args, "O!O!", &_TensorType, &grad, &_TensorType, &x)) return NULL;
    if (require_size(x, grad->size, "x") < 0) return NULL;

    _Tensor *result = alloc_result(grad->size);
    if (!result) return NULL;

    if (grad->size) gelu_bwd(grad->d_ptr, x->d_ptr, result->d_ptr, grad->size);
    return (PyObject *)result;
}

/* 0 if the gradient and optimizer state tensors all match the parameter's size,
 * else -1 with ValueError set. `m` may be NULL (SGD has one state buffer). */
static int check_param_state(const _Tensor *p, const _Tensor *g, const _Tensor *m, const _Tensor *v)
{
    if (require_size(g, p->size, "gradient") < 0) return -1;
    if (m && require_size(m, p->size, "first-moment state") < 0) return -1;
    return require_size(v, p->size, m ? "second-moment state" : "momentum state");
}

/* ---- fused optimizer steps (mutate p, m, v in place) ----
 * Adam and AdamW share these: plain Adam passes weight_decay = 0. */
static PyObject *_adam_step(PyObject *module, PyObject *args)
{
    _Tensor *p, *g, *m, *v;
    float lr, beta1, mb1, beta2, mb2, eps, bc1, bc2, weight_decay;
    if (!PyArg_ParseTuple(args, "O!O!O!O!fffffffff", &_TensorType, &p, &_TensorType, &g,
                          &_TensorType, &m, &_TensorType, &v, &lr, &beta1, &mb1, &beta2, &mb2, &eps,
                          &bc1, &bc2, &weight_decay))
        return NULL;
    if (check_param_state(p, g, m, v) < 0) return NULL;

    if (p->size)
        adam_step(p->d_ptr, g->d_ptr, m->d_ptr, v->d_ptr, p->size, lr, beta1, mb1, beta2, mb2, eps,
                  bc1, bc2, weight_decay);
    Py_RETURN_NONE;
}

static PyObject *_sgd_step(PyObject *module, PyObject *args)
{
    _Tensor *p, *g, *v;
    float lr, momentum;
    if (!PyArg_ParseTuple(args, "O!O!O!ff", &_TensorType, &p, &_TensorType, &g, &_TensorType, &v,
                          &lr, &momentum))
        return NULL;
    if (check_param_state(p, g, NULL, v) < 0) return NULL;

    if (p->size) sgd_step(p->d_ptr, g->d_ptr, v->d_ptr, p->size, lr, momentum);
    Py_RETURN_NONE;
}

/* ---- grouped (multi-tensor) optimizer steps ----
 * One kernel updates every parameter, so a 100-parameter model costs one launch
 * per step instead of 100. The kernel reads the per-parameter buffer pointers
 * from device arrays, which have to be uploaded -- but parameters and their
 * Adam state never move, and after warmup the caching allocator hands back the
 * same gradient buffers every step, so the upload is done once and then skipped
 * (a host-side memcmp) for the rest of training. That keeps the steady-state
 * cost at a single launch with no host<->device traffic and no sync. */
typedef struct {
    int nparams;
    float **hp, **hg, **hm, **hv;
    int *hs;
    float **dp, **dg, **dm, **dv;
    int *ds;
} _GroupCache;

static _GroupCache g_group = {0};

static void _group_free(_GroupCache *c)
{
    free(c->hp);
    free(c->hg);
    free(c->hm);
    free(c->hv);
    free(c->hs);
    free_memory(c->dp);
    free_memory(c->dg);
    free_memory(c->dm);
    free_memory(c->dv);
    free_memory(c->ds);
    memset(c, 0, sizeof(*c));
}

static int _group_alloc(_GroupCache *c, int nparams)
{
    size_t pb = (size_t)nparams * sizeof(float *);
    size_t sb = (size_t)nparams * sizeof(int);
    c->hp = (float **)malloc(pb);
    c->hg = (float **)malloc(pb);
    c->hm = (float **)malloc(pb);
    c->hv = (float **)malloc(pb);
    c->hs = (int *)malloc(sb);
    c->dp = (float **)alloc_memory(pb);
    c->dg = (float **)alloc_memory(pb);
    c->dm = (float **)alloc_memory(pb);
    c->dv = (float **)alloc_memory(pb);
    c->ds = (int *)alloc_memory(sb);
    if (!c->hp || !c->hg || !c->hm || !c->hv || !c->hs || !c->dp || !c->dg || !c->dm || !c->dv ||
        !c->ds) {
        _group_free(c);
        return 0;
    }
    c->nparams = nparams;
    /* Force the first upload: the freshly-malloc'd host mirrors are garbage, so
     * seed them with a value no real buffer can have. */
    for (int i = 0; i < nparams; i++) {
        c->hp[i] = c->hg[i] = c->hm[i] = c->hv[i] = (float *)(intptr_t)-1;
        c->hs[i] = -1;
    }
    return 1;
}

/* Refresh the device pointer/size arrays from `pl`/`gl`/`ml`/`vl`, uploading
 * only the ones that actually changed. Returns 0 (with a Python error set) on
 * failure; otherwise fills *max_size. */
static int _group_sync(PyObject *pl, PyObject *gl, PyObject *ml, PyObject *vl, int nparams,
                       int *max_size)
{
    if (g_group.nparams != nparams) {
        _group_free(&g_group);
        if (!_group_alloc(&g_group, nparams)) {
            PyErr_NoMemory();
            return 0;
        }
    }

    size_t pb = (size_t)nparams * sizeof(float *);
    size_t sb = (size_t)nparams * sizeof(int);
    float **hp = (float **)malloc(pb);
    float **hg = (float **)malloc(pb);
    float **hm = (float **)malloc(pb);
    float **hv = (float **)malloc(pb);
    int *hs = (int *)malloc(sb);
    if (!hp || !hg || !hm || !hv || !hs) {
        free(hp);
        free(hg);
        free(hm);
        free(hv);
        free(hs);
        PyErr_NoMemory();
        return 0;
    }

    int mx = 0;
    for (int i = 0; i < nparams; i++) {
        _Tensor *pp = (_Tensor *)PyList_GET_ITEM(pl, i);
        _Tensor *gg = (_Tensor *)PyList_GET_ITEM(gl, i);
        _Tensor *mm = (_Tensor *)PyList_GET_ITEM(ml, i);
        _Tensor *vv = (_Tensor *)PyList_GET_ITEM(vl, i);
        hp[i] = (float *)pp->d_ptr;
        hg[i] = (float *)gg->d_ptr;
        hm[i] = (float *)mm->d_ptr;
        hv[i] = (float *)vv->d_ptr;
        hs[i] = pp->size;
        if (pp->size > mx) mx = pp->size;
    }
    *max_size = mx;

    struct {
        void *host;
        void *cached;
        void *dev;
        size_t bytes;
    } arrays[5] = {
        {hp, g_group.hp, g_group.dp, pb}, {hg, g_group.hg, g_group.dg, pb},
        {hm, g_group.hm, g_group.dm, pb}, {hv, g_group.hv, g_group.dv, pb},
        {hs, g_group.hs, g_group.ds, sb},
    };
    for (int i = 0; i < 5; i++) {
        if (memcmp(arrays[i].host, arrays[i].cached, arrays[i].bytes) == 0) continue;
        /* copy_to_device runs on the default stream, so make sure nothing on
         * the ember stream is still reading the old contents. Only happens on
         * the first step (and after a reallocation). */
        sync_device();
        memcpy(arrays[i].cached, arrays[i].host, arrays[i].bytes);
        copy_to_device(arrays[i].dev, arrays[i].host, arrays[i].bytes);
    }

    free(hp);
    free(hg);
    free(hm);
    free(hv);
    free(hs);
    return 1;
}

static PyObject *_adam_step_group(PyObject *module, PyObject *args)
{
    PyObject *pl, *gl, *ml, *vl;
    float lr, beta1, mb1, beta2, mb2, eps, bc1, bc2, weight_decay;
    if (!PyArg_ParseTuple(args, "O!O!O!O!fffffffff", &PyList_Type, &pl, &PyList_Type, &gl,
                          &PyList_Type, &ml, &PyList_Type, &vl, &lr, &beta1, &mb1, &beta2, &mb2,
                          &eps, &bc1, &bc2, &weight_decay))
        return NULL;

    Py_ssize_t n = PyList_Size(pl);
    if (PyList_Size(gl) != n || PyList_Size(ml) != n || PyList_Size(vl) != n) {
        PyErr_SetString(PyExc_ValueError, "grouped optimizer: parameter list lengths differ");
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *items[4] = {PyList_GET_ITEM(pl, i), PyList_GET_ITEM(gl, i),
                              PyList_GET_ITEM(ml, i), PyList_GET_ITEM(vl, i)};
        for (int j = 0; j < 4; j++) {
            if (!PyObject_TypeCheck(items[j], &_TensorType)) {
                PyErr_SetString(PyExc_TypeError, "grouped optimizer: lists must hold _Tensor");
                return NULL;
            }
        }
        if (check_param_state((_Tensor *)items[0], (_Tensor *)items[1], (_Tensor *)items[2],
                              (_Tensor *)items[3]) < 0)
            return NULL;
    }
    int nparams = (int)n;
    if (nparams == 0) Py_RETURN_NONE;

    int max_size = 0;
    if (!_group_sync(pl, gl, ml, vl, nparams, &max_size)) return NULL;
    if (max_size == 0) Py_RETURN_NONE;  // only empty parameters

    adam_step_group(g_group.dp, g_group.dg, g_group.dm, g_group.dv, g_group.ds, nparams, max_size,
                    lr, beta1, mb1, beta2, mb2, eps, bc1, bc2, weight_decay);
    Py_RETURN_NONE;
}

/* ---- capturable Adam (device-side step counter + bias correction) ---- */
static PyObject *_adam_bias_update(PyObject *module, PyObject *args)
{
    _Tensor *t, *bc;
    float beta1, beta2;
    if (!PyArg_ParseTuple(args, "O!O!ff", &_TensorType, &t, &_TensorType, &bc, &beta1, &beta2))
        return NULL;
    if (require_size(t, 1, "step counter") < 0 || require_size(bc, 2, "bias corrections") < 0)
        return NULL;

    adam_bias_update(t->d_ptr, bc->d_ptr, beta1, beta2);
    Py_RETURN_NONE;
}

static PyObject *_adam_step_dev(PyObject *module, PyObject *args)
{
    _Tensor *p, *g, *m, *v, *bc;
    float lr, beta1, mb1, beta2, mb2, eps, weight_decay;
    if (!PyArg_ParseTuple(args, "O!O!O!O!ffffffO!f", &_TensorType, &p, &_TensorType, &g,
                          &_TensorType, &m, &_TensorType, &v, &lr, &beta1, &mb1, &beta2, &mb2, &eps,
                          &_TensorType, &bc, &weight_decay))
        return NULL;
    if (check_param_state(p, g, m, v) < 0 || require_size(bc, 2, "bias corrections") < 0)
        return NULL;

    if (p->size)
        adam_step_dev(p->d_ptr, g->d_ptr, m->d_ptr, v->d_ptr, p->size, lr, beta1, mb1, beta2, mb2,
                      eps, bc->d_ptr, weight_decay);
    Py_RETURN_NONE;
}

/* ---- device / CUDA-graph control ---- */
static PyObject *_sync(PyObject *module, PyObject *args)
{
    sync_device();
    Py_RETURN_NONE;
}

static PyObject *_empty_cache(PyObject *module, PyObject *args)
{
    empty_device_cache();
    Py_RETURN_NONE;
}

static PyObject *_memory_allocated(PyObject *module, PyObject *args)
{
    return PyLong_FromSize_t(device_memory_in_use());
}

static PyObject *_set_matmul_tf32(PyObject *module, PyObject *args)
{
    int enabled;
    if (!PyArg_ParseTuple(args, "p", &enabled)) return NULL;
    set_matmul_tf32(enabled);
    Py_RETURN_NONE;
}

static PyObject *_get_matmul_tf32(PyObject *module, PyObject *args)
{
    return PyBool_FromLong(get_matmul_tf32());
}

static PyObject *_begin_capture(PyObject *module, PyObject *args)
{
    begin_capture();
    Py_RETURN_NONE;
}

static PyObject *_end_capture(PyObject *module, PyObject *args)
{
    return PyLong_FromVoidPtr(end_capture());
}

static PyObject *_graph_launch(PyObject *module, PyObject *args)
{
    PyObject *handle;
    if (!PyArg_ParseTuple(args, "O", &handle)) return NULL;
    graph_launch(PyLong_AsVoidPtr(handle));
    Py_RETURN_NONE;
}

static PyObject *_graph_destroy(PyObject *module, PyObject *args)
{
    PyObject *handle;
    if (!PyArg_ParseTuple(args, "O", &handle)) return NULL;
    graph_destroy(PyLong_AsVoidPtr(handle));
    Py_RETURN_NONE;
}

/* ---- type & module definitions ---- */
static PyMethodDef _Tensor_instance_methods[] = {
    {"_copy_from_numpy", (PyCFunction)_Tensor_copy_from_numpy, METH_VARARGS,
     "Update the device buffer in place from a float32 numpy array"},
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
#define EMBER_INPLACE_OP(name, expr) OP_METHOD(_##name),
#define EMBER_INPLACE_SCALAR_OP(name, expr) OP_METHOD(_##name##_scalar),
#include "operators.def"

    /* non-element-wise operators */
    OP_METHOD(_matmul),
    OP_METHOD(_matmul_bias),
    OP_METHOD(_matmul_batched),
    OP_METHOD(_transpose),
    OP_METHOD(_from_numpy),
    OP_METHOD(_sum),
    OP_METHOD(_sum_axis),
    OP_METHOD(_max_axis),
    OP_METHOD(_gelu_bwd),
    OP_METHOD(_adam_step),
    OP_METHOD(_sgd_step),
    OP_METHOD(_adam_step_group),
    OP_METHOD(_adam_bias_update),
    OP_METHOD(_adam_step_dev),

    // device / CUDA-graph control
    {"_sync", (PyCFunction)_sync, METH_NOARGS, "Synchronize the device"},
    {"_empty_cache", (PyCFunction)_empty_cache, METH_NOARGS, "Release the cached device pool"},
    {"_memory_allocated", (PyCFunction)_memory_allocated, METH_NOARGS,
     "Bytes of device memory held by live tensors"},
    {"_set_matmul_tf32", (PyCFunction)_set_matmul_tf32, METH_VARARGS,
     "Enable/disable TF32 tensor cores for cuBLAS GEMMs"},
    {"_get_matmul_tf32", (PyCFunction)_get_matmul_tf32, METH_NOARGS,
     "Whether TF32 matmuls are enabled"},
    {"_begin_capture", (PyCFunction)_begin_capture, METH_NOARGS, "Begin CUDA-graph capture"},
    {"_end_capture", (PyCFunction)_end_capture, METH_NOARGS, "End capture; return graph handle"},
    {"_graph_launch", (PyCFunction)_graph_launch, METH_VARARGS, "Replay a captured graph"},
    {"_graph_destroy", (PyCFunction)_graph_destroy, METH_VARARGS, "Destroy a captured graph"},

    {NULL}};

PyTypeObject _TensorType = {
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
    import_array();  // NumPy C API (used by _to_np)
    if (PyType_Ready(&_TensorType) < 0) return NULL;

    PyObject *m = PyModule_Create(&tensor_module);
    if (!m) return NULL;

    Py_INCREF(&_TensorType);
    if (PyModule_AddObject(m, "_Tensor", (PyObject *)&_TensorType) < 0) {
        Py_DECREF(&_TensorType);
        Py_DECREF(m);
        return NULL;
    }

    /* transformer building blocks (transformer_api.c), random fills (random_api.c) */
    if (PyModule_AddFunctions(m, transformer_methods) < 0 ||
        PyModule_AddFunctions(m, random_methods) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
