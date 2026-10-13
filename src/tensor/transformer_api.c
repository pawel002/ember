/* Python bindings for the transformer building blocks declared in
 * transformer.h. Part of the _tensor module (see tensor_module.h).
 *
 * Every shape argument is checked against the tensors it describes before a
 * kernel runs (see tensor_helpers.h). These ops need non-empty inputs: an empty
 * dimension raises ValueError rather than launching an empty kernel. */

#include <Python.h>
#include <string.h>

#include "../core/memory.h"
#include "tensor_helpers.h"
#include "tensor_module.h"
#include "transformer.h"

/* 0 if every dimension in `dims` is positive, else -1 with ValueError set. */
static int require_positive(const char *op, int count, const int *dims)
{
    for (int i = 0; i < count; i++) {
        if (dims[i] <= 0) {
            PyErr_Format(PyExc_ValueError, "%s: dimensions must be positive", op);
            return -1;
        }
    }
    return 0;
}

/* ---- fused softmax over an axis ---- */
static PyObject *_softmax_fwd(PyObject *module, PyObject *args)
{
    _Tensor *a;
    PyObject *shape;
    int axis;
    if (!PyArg_ParseTuple(args, "O!O!i", &_TensorType, &a, &PyTuple_Type, &shape, &axis))
        return NULL;

    int outer, axis_dim, inner;
    if (parse_axis(shape, axis, a->size, &outer, &axis_dim, &inner) < 0) return NULL;

    _Tensor *result = alloc_result(a->size);
    if (!result) return NULL;
    if (a->size) softmax_axis(a->d_ptr, result->d_ptr, outer, inner, axis_dim);
    return (PyObject *)result;
}

static PyObject *_softmax_bwd(PyObject *module, PyObject *args)
{
    _Tensor *dout, *y;
    PyObject *shape;
    int axis;
    if (!PyArg_ParseTuple(args, "O!O!O!i", &_TensorType, &dout, &_TensorType, &y, &PyTuple_Type,
                          &shape, &axis))
        return NULL;

    int outer, axis_dim, inner;
    if (parse_axis(shape, axis, dout->size, &outer, &axis_dim, &inner) < 0) return NULL;
    if (require_size(y, dout->size, "y") < 0) return NULL;

    _Tensor *result = alloc_result(dout->size);
    if (!result) return NULL;
    if (dout->size) softmax_axis_bwd(dout->d_ptr, y->d_ptr, result->d_ptr, outer, inner, axis_dim);
    return (PyObject *)result;
}

/* ---- attention softmax over contiguous rows (optional causal mask) ---- */
static PyObject *_softmax_rows(PyObject *module, PyObject *args)
{
    _Tensor *a;
    int rows, D;
    if (!PyArg_ParseTuple(args, "O!ii", &_TensorType, &a, &rows, &D)) return NULL;
    int dims[] = {rows, D};
    if (require_positive("softmax_rows", 2, dims) < 0 ||
        require_size(a, (long long)rows * D, "input") < 0)
        return NULL;

    _Tensor *result = alloc_result(a->size);
    if (!result) return NULL;
    softmax_rows(a->d_ptr, result->d_ptr, rows, D);
    return (PyObject *)result;
}

static PyObject *_softmax_rows_causal(PyObject *module, PyObject *args)
{
    _Tensor *a;
    int rows, D, sq;
    if (!PyArg_ParseTuple(args, "O!iii", &_TensorType, &a, &rows, &D, &sq)) return NULL;
    int dims[] = {rows, D, sq};
    if (require_positive("softmax_rows_causal", 3, dims) < 0 ||
        require_size(a, (long long)rows * D, "input") < 0)
        return NULL;

    _Tensor *result = alloc_result(a->size);
    if (!result) return NULL;
    softmax_rows_causal(a->d_ptr, result->d_ptr, rows, D, sq);
    return (PyObject *)result;
}

/* ---- LayerNorm (returns (out, mean, rstd)) ---- */
static PyObject *_layernorm_fwd(PyObject *module, PyObject *args)
{
    _Tensor *x, *gamma, *beta;
    int N, D;
    float eps;
    if (!PyArg_ParseTuple(args, "O!O!O!iif", &_TensorType, &x, &_TensorType, &gamma, &_TensorType,
                          &beta, &N, &D, &eps))
        return NULL;
    int dims[] = {N, D};
    if (require_positive("layernorm", 2, dims) < 0 || require_size(x, (long long)N * D, "x") < 0 ||
        require_size(gamma, D, "gamma") < 0 || require_size(beta, D, "beta") < 0)
        return NULL;

    _Tensor *out = alloc_result(x->size);
    _Tensor *mean = alloc_result(N);
    _Tensor *rstd = alloc_result(N);
    if (!out || !mean || !rstd) {
        Py_XDECREF(out);
        Py_XDECREF(mean);
        Py_XDECREF(rstd);
        return NULL;
    }
    layernorm_fwd(x->d_ptr, gamma->d_ptr, beta->d_ptr, out->d_ptr, mean->d_ptr, rstd->d_ptr, N, D,
                  eps);
    return Py_BuildValue("NNN", out, mean, rstd);
}

/* ---- LayerNorm backward (returns (dx, dgamma, dbeta)) ---- */
static PyObject *_layernorm_bwd(PyObject *module, PyObject *args)
{
    _Tensor *dout, *x, *gamma, *mean, *rstd;
    int N, D;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!ii", &_TensorType, &dout, &_TensorType, &x, &_TensorType,
                          &gamma, &_TensorType, &mean, &_TensorType, &rstd, &N, &D))
        return NULL;
    int dims[] = {N, D};
    if (require_positive("layernorm", 2, dims) < 0 ||
        require_size(dout, (long long)N * D, "dout") < 0 ||
        require_size(x, (long long)N * D, "x") < 0 || require_size(gamma, D, "gamma") < 0 ||
        require_size(mean, N, "mean") < 0 || require_size(rstd, N, "rstd") < 0)
        return NULL;

    _Tensor *dx = alloc_result(x->size);
    _Tensor *dgamma = alloc_result(D);
    _Tensor *dbeta = alloc_result(D);
    if (!dx || !dgamma || !dbeta) {
        Py_XDECREF(dx);
        Py_XDECREF(dgamma);
        Py_XDECREF(dbeta);
        return NULL;
    }
    layernorm_bwd(dout->d_ptr, x->d_ptr, gamma->d_ptr, mean->d_ptr, rstd->d_ptr, dx->d_ptr,
                  dgamma->d_ptr, dbeta->d_ptr, N, D);
    return Py_BuildValue("NNN", dx, dgamma, dbeta);
}

/* ---- (0,1,2,3)->(0,2,1,3) permutation ---- */
static PyObject *_permute_0213(PyObject *module, PyObject *args)
{
    _Tensor *x;
    int d0, d1, d2, d3;
    if (!PyArg_ParseTuple(args, "O!iiii", &_TensorType, &x, &d0, &d1, &d2, &d3)) return NULL;
    if (d0 < 0 || d1 < 0 || d2 < 0 || d3 < 0) {
        PyErr_SetString(PyExc_ValueError, "permute_0213: dimensions cannot be negative");
        return NULL;
    }
    if (require_size(x, (long long)d0 * d1 * d2 * d3, "input") < 0) return NULL;

    _Tensor *result = alloc_result(x->size);
    if (!result) return NULL;
    if (x->size) permute_0213(x->d_ptr, result->d_ptr, d0, d1, d2, d3);
    return (PyObject *)result;
}

/* ---- batched GEMM: C = alpha * opA(A) @ opB(B), per batch ---- */
static PyObject *_bmm(PyObject *module, PyObject *args)
{
    _Tensor *a, *b;
    int batch, n, m, k, transA, transB;
    float alpha;
    if (!PyArg_ParseTuple(args, "O!O!iiiiiif", &_TensorType, &a, &_TensorType, &b, &batch, &n, &m,
                          &k, &transA, &transB, &alpha))
        return NULL;
    int dims[] = {batch, n, m, k};
    if (require_positive("bmm", 4, dims) < 0 ||
        require_size(a, (long long)batch * n * k, "A") < 0 ||
        require_size(b, (long long)batch * k * m, "B") < 0)
        return NULL;
    int size = checked_size((long long)batch * n * m);
    if (size < 0) return NULL;

    _Tensor *result = alloc_result(size);
    if (!result) return NULL;
    bmm(a->d_ptr, b->d_ptr, result->d_ptr, batch, n, m, k, transA, transB, alpha);
    return (PyObject *)result;
}

/* ---- embedding ----
 * idx is a C-contiguous int32 array of row indices; each must be in
 * [0, vocab). The indices are on the host anyway, so the range check is a
 * cheap loop that keeps a bad token id from reading (forward) or atomically
 * writing (backward) outside the table. */
static int parse_indices(PyObject *obj, int vocab, Py_buffer *view, int *n_idx)
{
    if (PyObject_GetBuffer(obj, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        PyErr_SetString(PyExc_TypeError, "indices must be a C-contiguous int32 array");
        return -1;
    }
    /* struct-module format of a 4-byte signed int, optionally with a byte-order
     * prefix ("i", "<i", "=i", ...; "l" where long is 4 bytes). */
    const char *fmt = view->format ? view->format : "B";
    if (strchr("@=<>!", fmt[0])) fmt++;
    if (view->itemsize != 4 || (strcmp(fmt, "i") != 0 && strcmp(fmt, "l") != 0)) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_TypeError, "indices must be int32");
        return -1;
    }

    Py_ssize_t n = view->len / view->itemsize;
    if (checked_size(n) < 0) {
        PyBuffer_Release(view);
        return -1;
    }
    const int *idx = (const int *)view->buf;
    for (Py_ssize_t i = 0; i < n; i++) {
        if (idx[i] < 0 || idx[i] >= vocab) {
            PyBuffer_Release(view);
            PyErr_Format(PyExc_IndexError, "index %d is out of range for %d embeddings", idx[i],
                         vocab);
            return -1;
        }
    }
    *n_idx = (int)n;
    return 0;
}

/* Upload the indices, run `fwd` or `bwd`, and free the device copy. */
static PyObject *embedding_call(_Tensor *src, PyObject *idx_obj, int vocab, int dim, int backward)
{
    if (dim <= 0 || vocab <= 0) {
        PyErr_SetString(PyExc_ValueError, "embedding: vocab and dim must be positive");
        return NULL;
    }
    Py_buffer view;
    int n_idx;
    if (parse_indices(idx_obj, vocab, &view, &n_idx) < 0) return NULL;

    long long out_size = backward ? (long long)vocab * dim : (long long)n_idx * dim;
    if ((backward && require_size(src, (long long)n_idx * dim, "dout") < 0) ||
        checked_size(out_size) < 0) {
        PyBuffer_Release(&view);
        return NULL;
    }

    _Tensor *result = alloc_result((int)out_size);
    if (!result) {
        PyBuffer_Release(&view);
        return NULL;
    }
    if (n_idx == 0) {
        PyBuffer_Release(&view);
        if (backward) {
            Py_DECREF(result);
            PyErr_SetString(PyExc_ValueError, "embedding backward needs at least one index");
            return NULL;
        }
        return (PyObject *)result;
    }

    int *d_idx = (int *)alloc_memory((size_t)n_idx * sizeof(int));
    if (!d_idx) {
        Py_DECREF(result);
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }
    copy_to_device(d_idx, view.buf, (size_t)n_idx * sizeof(int));
    if (backward)
        embedding_bwd(src->d_ptr, d_idx, result->d_ptr, n_idx, dim, vocab);
    else
        embedding_fwd(src->d_ptr, d_idx, result->d_ptr, n_idx, dim);
    sync_device();  // done with the index buffer before it returns to the pool
    free_memory(d_idx);
    PyBuffer_Release(&view);
    return (PyObject *)result;
}

static PyObject *_embedding_fwd(PyObject *module, PyObject *args)
{
    _Tensor *weight;
    PyObject *idx_obj;
    int dim;
    if (!PyArg_ParseTuple(args, "O!Oi", &_TensorType, &weight, &idx_obj, &dim)) return NULL;
    if (dim <= 0 || weight->size % dim != 0) {
        PyErr_Format(PyExc_ValueError, "embedding table of %d elements is not a multiple of dim %d",
                     weight->size, dim);
        return NULL;
    }
    return embedding_call(weight, idx_obj, weight->size / dim, dim, 0);
}

static PyObject *_embedding_bwd(PyObject *module, PyObject *args)
{
    _Tensor *dout;
    PyObject *idx_obj;
    int vocab, dim;
    if (!PyArg_ParseTuple(args, "O!Oii", &_TensorType, &dout, &idx_obj, &vocab, &dim)) return NULL;
    return embedding_call(dout, idx_obj, vocab, dim, 1);
}

/* ---- fused attention ----
 * _attention_fwd(q, k, v, batch, sq, sk, dh, scale, causal, nheads, sseq)
 *     -> (out, lse)
 * _attention_bwd(dout, q, k, v, out, lse, batch, sq, sk, dh, scale, causal,
 *                nheads, sseq) -> (dq, dk, dv)
 * nheads/sseq describe the interleaving of q/k/v/out (see transformer.h); the
 * results follow the same layout as their inputs. */
static PyObject *_attention_supported(PyObject *module, PyObject *args)
{
    int dh;
    if (!PyArg_ParseTuple(args, "i", &dh)) return NULL;
    return PyBool_FromLong(attention_supported(dh));
}

/* Check the shared attention arguments; 0 or -1 with a Python error set. */
static int check_attention(int batch, int sq, int sk, int dh, int nheads, int sseq,
                           const _Tensor *q, const _Tensor *k, const _Tensor *v)
{
    int dims[] = {batch, sq, sk, dh, nheads};
    if (require_positive("attention", 5, dims) < 0) return -1;
    if (!attention_supported(dh)) {
        PyErr_Format(PyExc_ValueError, "attention: no fused kernel for head_dim %d", dh);
        return -1;
    }
    if (batch % nheads != 0 || sseq != nheads * dh) {
        PyErr_SetString(PyExc_ValueError,
                        "attention: layout needs batch % nheads == 0 and sseq == nheads * dh");
        return -1;
    }
    if (require_size(q, (long long)batch * sq * dh, "q") < 0 ||
        require_size(k, (long long)batch * sk * dh, "k") < 0 ||
        require_size(v, (long long)batch * sk * dh, "v") < 0)
        return -1;
    return 0;
}

static PyObject *_attention_fwd(PyObject *module, PyObject *args)
{
    _Tensor *q, *k, *v;
    int batch, sq, sk, dh, causal, nheads, sseq;
    float scale;
    if (!PyArg_ParseTuple(args, "O!O!O!iiiifiii", &_TensorType, &q, &_TensorType, &k, &_TensorType,
                          &v, &batch, &sq, &sk, &dh, &scale, &causal, &nheads, &sseq))
        return NULL;
    if (check_attention(batch, sq, sk, dh, nheads, sseq, q, k, v) < 0) return NULL;

    _Tensor *out = alloc_result(q->size);
    _Tensor *lse = alloc_result(batch * sq);
    if (!out || !lse) {
        Py_XDECREF(out);
        Py_XDECREF(lse);
        return NULL;
    }

    attention_fwd(q->d_ptr, k->d_ptr, v->d_ptr, out->d_ptr, lse->d_ptr, batch, sq, sk, dh, scale,
                  causal, nheads, sseq);
    return Py_BuildValue("(NN)", (PyObject *)out, (PyObject *)lse);
}

static PyObject *_attention_bwd(PyObject *module, PyObject *args)
{
    _Tensor *dout, *q, *k, *v, *out, *lse;
    int batch, sq, sk, dh, causal, nheads, sseq;
    float scale;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iiiifiii", &_TensorType, &dout, &_TensorType, &q,
                          &_TensorType, &k, &_TensorType, &v, &_TensorType, &out, &_TensorType,
                          &lse, &batch, &sq, &sk, &dh, &scale, &causal, &nheads, &sseq))
        return NULL;
    if (check_attention(batch, sq, sk, dh, nheads, sseq, q, k, v) < 0 ||
        require_size(dout, q->size, "dout") < 0 || require_size(out, q->size, "out") < 0 ||
        require_size(lse, (long long)batch * sq, "lse") < 0)
        return NULL;

    _Tensor *dq = alloc_result(q->size);
    _Tensor *dk = alloc_result(k->size);
    _Tensor *dv = alloc_result(v->size);
    float *rowdot = (float *)alloc_memory((size_t)batch * sq * sizeof(float));  // scratch
    if (!dq || !dk || !dv || !rowdot) {
        Py_XDECREF(dq);
        Py_XDECREF(dk);
        Py_XDECREF(dv);
        free_memory(rowdot);
        return dq && dk && dv ? PyErr_NoMemory() : NULL;
    }

    attention_bwd(dout->d_ptr, q->d_ptr, k->d_ptr, v->d_ptr, out->d_ptr, lse->d_ptr, rowdot,
                  dq->d_ptr, dk->d_ptr, dv->d_ptr, batch, sq, sk, dh, scale, causal, nheads, sseq);
    // Single-stream ordering: the block cannot be reused before the kernels ran.
    free_memory(rowdot);
    return Py_BuildValue("(NNN)", (PyObject *)dq, (PyObject *)dk, (PyObject *)dv);
}

PyMethodDef transformer_methods[] = {
    OP_METHOD(_softmax_fwd),         OP_METHOD(_softmax_bwd),
    OP_METHOD(_softmax_rows),        OP_METHOD(_softmax_rows_causal),
    OP_METHOD(_layernorm_fwd),       OP_METHOD(_layernorm_bwd),
    OP_METHOD(_permute_0213),        OP_METHOD(_bmm),
    OP_METHOD(_embedding_fwd),       OP_METHOD(_embedding_bwd),
    OP_METHOD(_attention_supported), OP_METHOD(_attention_fwd),
    OP_METHOD(_attention_bwd),       {NULL}};
