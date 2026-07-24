/* Python bindings for the transformer building blocks declared in
 * transformer.h. Textually included into tensor_api.c (after alloc_result and
 * _TensorType are defined) so these ops join the _tensor module and share its
 * CUDA stream + caching allocator. Not compiled on its own. */

/* ---- fused softmax over an axis ---- */
static PyObject *_softmax_fwd(PyObject *module, PyObject *args)
{
    _Tensor *a;
    PyObject *shape_obj;
    int axis;
    if (!PyArg_ParseTuple(args, "O!O!i", &_TensorType, &a, &PyTuple_Type, &shape_obj, &axis))
        return NULL;

    int ndim, *shape;
    if (tuple_to_array(shape_obj, &shape, &ndim) < 0) return NULL;
    int outer = sum_axis_product(shape, 0, axis);
    int axis_dim = shape[axis];
    int inner = sum_axis_product(shape, axis + 1, ndim);
    free(shape);

    _Tensor *result = alloc_result(a->size);
    if (!result) return NULL;
    softmax_axis(a->d_ptr, result->d_ptr, outer, inner, axis_dim);
    return (PyObject *)result;
}

static PyObject *_softmax_bwd(PyObject *module, PyObject *args)
{
    _Tensor *dout, *y;
    PyObject *shape_obj;
    int axis;
    if (!PyArg_ParseTuple(args, "O!O!O!i", &_TensorType, &dout, &_TensorType, &y, &PyTuple_Type,
                          &shape_obj, &axis))
        return NULL;

    int ndim, *shape;
    if (tuple_to_array(shape_obj, &shape, &ndim) < 0) return NULL;
    int outer = sum_axis_product(shape, 0, axis);
    int axis_dim = shape[axis];
    int inner = sum_axis_product(shape, axis + 1, ndim);
    free(shape);

    _Tensor *result = alloc_result(dout->size);
    if (!result) return NULL;
    softmax_axis_bwd(dout->d_ptr, y->d_ptr, result->d_ptr, outer, inner, axis_dim);
    return (PyObject *)result;
}

/* ---- attention softmax over contiguous rows (optional causal mask) ---- */
static PyObject *_softmax_rows(PyObject *module, PyObject *args)
{
    _Tensor *a;
    int rows, D;
    if (!PyArg_ParseTuple(args, "O!ii", &_TensorType, &a, &rows, &D)) return NULL;

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

    _Tensor *out = alloc_result(N * D);
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

    _Tensor *dx = alloc_result(N * D);
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

    _Tensor *result = alloc_result(d0 * d1 * d2 * d3);
    if (!result) return NULL;
    permute_0213(x->d_ptr, result->d_ptr, d0, d1, d2, d3);
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

    _Tensor *result = alloc_result(batch * n * m);
    if (!result) return NULL;
    bmm(a->d_ptr, b->d_ptr, result->d_ptr, batch, n, m, k, transA, transB, alpha);
    return (PyObject *)result;
}

/* ---- embedding: idx is a C-contiguous int32 numpy array ---- */
static int parse_int_indices(PyObject *obj, Py_buffer *view, int *n_idx)
{
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (PyObject_GetBuffer(obj, view, flags) < 0) {
        PyErr_SetString(PyExc_TypeError, "indices must be a C-contiguous int32 array");
        return -1;
    }
    if (view->itemsize != 4) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_TypeError, "indices must be 4-byte integers (int32)");
        return -1;
    }
    *n_idx = (int)(view->len / view->itemsize);
    return 0;
}

static PyObject *_embedding_fwd(PyObject *module, PyObject *args)
{
    _Tensor *weight;
    PyObject *idx_obj;
    int dim;
    if (!PyArg_ParseTuple(args, "O!Oi", &_TensorType, &weight, &idx_obj, &dim)) return NULL;

    Py_buffer view;
    int n_idx;
    if (parse_int_indices(idx_obj, &view, &n_idx) < 0) return NULL;

    int *d_idx = (int *)alloc_memory((size_t)n_idx * sizeof(int));
    _Tensor *result = alloc_result(n_idx * dim);
    if (!d_idx || !result) {
        if (d_idx) free_memory(d_idx);
        Py_XDECREF(result);
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }
    copy_to_device(d_idx, view.buf, (size_t)n_idx * sizeof(int));
    embedding_fwd(weight->d_ptr, d_idx, result->d_ptr, n_idx, dim);
    sync_device();  // done with the index buffer before it returns to the pool
    free_memory(d_idx);
    PyBuffer_Release(&view);
    return (PyObject *)result;
}

static PyObject *_embedding_bwd(PyObject *module, PyObject *args)
{
    _Tensor *dout;
    PyObject *idx_obj;
    int vocab, dim;
    if (!PyArg_ParseTuple(args, "O!Oii", &_TensorType, &dout, &idx_obj, &vocab, &dim)) return NULL;

    Py_buffer view;
    int n_idx;
    if (parse_int_indices(idx_obj, &view, &n_idx) < 0) return NULL;

    int *d_idx = (int *)alloc_memory((size_t)n_idx * sizeof(int));
    _Tensor *result = alloc_result(vocab * dim);
    if (!d_idx || !result) {
        if (d_idx) free_memory(d_idx);
        Py_XDECREF(result);
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }
    copy_to_device(d_idx, view.buf, (size_t)n_idx * sizeof(int));
    embedding_bwd(dout->d_ptr, d_idx, result->d_ptr, n_idx, dim, vocab);
    sync_device();
    free_memory(d_idx);
    PyBuffer_Release(&view);
    return (PyObject *)result;
}

#define EMBER_TRANSFORMER_METHODS                                                              \
    OP_METHOD(_softmax_fwd), OP_METHOD(_softmax_bwd), OP_METHOD(_softmax_rows),                \
        OP_METHOD(_softmax_rows_causal), OP_METHOD(_layernorm_fwd), OP_METHOD(_layernorm_bwd), \
        OP_METHOD(_permute_0213), OP_METHOD(_bmm), OP_METHOD(_embedding_fwd),                  \
        OP_METHOD(_embedding_bwd),
