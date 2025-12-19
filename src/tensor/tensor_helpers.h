#include <Python.h>

static PyObject* build_nested_list(float* data, long* dims, int ndim, int current_dim, int* offset) {
    long size = dims[current_dim];
    PyObject* list = PyList_New(size);
    if (!list) return NULL;

    if (current_dim == ndim - 1) {
        for (long i = 0; i < size; i++) {
            PyObject* num = PyFloat_FromDouble(data[*offset]);
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
            PyObject* sublist = build_nested_list(data, dims, ndim, current_dim + 1, offset);
            if (!sublist) {
                Py_DECREF(list);
                return NULL;
            }
            PyList_SetItem(list, i, sublist);
        }
    }

    return list;
}