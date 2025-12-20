#include <Python.h>
#include <structmember.h>

#include "tensor_helpers.h"
#include "ops.h" 
#include "../core/memory.h"

typedef struct {
    PyObject_HEAD
    void* d_ptr;
    int size;
} _Tensor;

static void _Tensor_dealloc(_Tensor* self) {
    if (self->d_ptr) {
        free_memory(self->d_ptr);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int _Tensor_init(_Tensor* self, PyObject* args, PyObject* kwds) {
    int size = 0;
    if (!PyArg_ParseTuple(args, "i", &size)) {
        return -1;
    }
    
    self->size = size;
    self->d_ptr = alloc_memory(size * sizeof(float));
    
    if (self->d_ptr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate GPU memory");
        return -1;
    }
    return 0;
}

static PyObject* _Tensor_add(_Tensor* self, PyObject* args) {
    PyObject* other_obj;
    if (!PyArg_ParseTuple(args, "O", &other_obj)) return NULL;

    _Tensor* other = (_Tensor*)other_obj;

    if (self->size != other->size) {
        PyErr_SetString(PyExc_ValueError, "Backend size mismatch");
        return NULL;
    }

    _Tensor* result = (_Tensor*)Py_TYPE(self)->tp_alloc(Py_TYPE(self), 0);
    if (!result) return NULL;
    
    result->size = self->size;
    result->d_ptr = alloc_memory(self->size * sizeof(float));

    add(self->d_ptr, other->d_ptr, result->d_ptr, self->size);

    return (PyObject*)result;
}

static PyObject* _Tensor_subtract(_Tensor* self, PyObject* args) {
    PyObject* other_obj;
    if (!PyArg_ParseTuple(args, "O", &other_obj)) return NULL;

    _Tensor* other = (_Tensor*)other_obj;

    if (self->size != other->size) {
        PyErr_SetString(PyExc_ValueError, "Backend size mismatch");
        return NULL;
    }

    _Tensor* result = (_Tensor*)Py_TYPE(self)->tp_alloc(Py_TYPE(self), 0);
    if (!result) return NULL;
    
    result->size = self->size;
    result->d_ptr = alloc_memory(self->size * sizeof(float));

    subtract(self->d_ptr, other->d_ptr, result->d_ptr, self->size);

    return (PyObject*)result;
}

static PyObject* _Tensor_multiply_elementwise(_Tensor* self, PyObject* args) {
    PyObject* other_obj;
    if (!PyArg_ParseTuple(args, "O", &other_obj)) return NULL;

    _Tensor* other = (_Tensor*)other_obj;

    if (self->size != other->size) {
        PyErr_SetString(PyExc_ValueError, "Backend size mismatch");
        return NULL;
    }

    _Tensor* result = (_Tensor*)Py_TYPE(self)->tp_alloc(Py_TYPE(self), 0);
    if (!result) return NULL;
    
    result->size = self->size;
    result->d_ptr = alloc_memory(self->size * sizeof(float));

    multiply_elementwise(self->d_ptr, other->d_ptr, result->d_ptr, self->size);

    return (PyObject*)result;
}

static PyObject* _Tensor_copy_from_list(_Tensor* self, PyObject* args) {
    PyObject* py_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &py_list)) return NULL;

    float* temp_host = (float*)malloc(self->size * sizeof(float));
    
    for (int i=0; i<self->size; i++) {
        PyObject* item = PyList_GetItem(py_list, i);
        temp_host[i] = (float)PyFloat_AsDouble(item);
    }

    copy_to_device(self->d_ptr, temp_host, self->size * sizeof(float));
    free(temp_host);
    
    Py_RETURN_NONE;
}

static PyObject* _Tensor_to_list(_Tensor* self, PyObject* args) {
    PyObject* shape_tuple;

    if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &shape_tuple)) {
        return NULL;
    }

    Py_ssize_t ndim = PyTuple_Size(shape_tuple);
    long* c_dims = (long*)malloc(ndim * sizeof(long));
    long total_elements = 1;

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PyTuple_GetItem(shape_tuple, i);
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

    float* temp_host = (float*)malloc(self->size * sizeof(float));
    if (!temp_host) {
        free(c_dims);
        return PyErr_NoMemory();
    }

    copy_from_device(temp_host, self->d_ptr, self->size * sizeof(float));

    int offset = 0;
    PyObject* result = build_nested_list(temp_host, c_dims, (int)ndim, 0, &offset);

    free(temp_host);
    free(c_dims);

    return result;
}

static PyMemberDef _Tensor_members[] = {
    {"size", T_INT, offsetof(_Tensor, size), READONLY, "Size of the tensor"},
    {NULL}
};

static PyMethodDef _Tensor_methods[] = {
    {"_add", (PyCFunction)_Tensor_add, METH_VARARGS, "Low level add"},
    {"_subtract", (PyCFunction)_Tensor_subtract, METH_VARARGS, "Low level subtraction"},
    {"_multiply_elementwise", (PyCFunction)_Tensor_multiply_elementwise, METH_VARARGS, "Low level multiplication"},
    {"copy_from_list", (PyCFunction)_Tensor_copy_from_list, METH_VARARGS, "Load data"},
    {"to_list", (PyCFunction)_Tensor_to_list, METH_VARARGS, "Read data"},
    {NULL}
};

static PyTypeObject _TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ember._core._tensor._Tensor",
    .tp_basicsize = sizeof(_Tensor),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = _Tensor_members,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)_Tensor_init,
    .tp_dealloc = (destructor)_Tensor_dealloc,
    .tp_methods = _Tensor_methods,
};

static PyModuleDef tensor_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "ember._core._tensor",
    .m_doc = "Ember Tensor backend",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__tensor(void) {
    if (PyType_Ready(&_TensorType) < 0) return NULL;

    PyObject* m = PyModule_Create(&tensor_module);
    if (!m) return NULL;

    Py_INCREF(&_TensorType);
    if (PyModule_AddObject(m, "_Tensor", (PyObject*)&_TensorType) < 0) {
        Py_DECREF(&_TensorType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}