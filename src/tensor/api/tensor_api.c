#include <Python.h>
#include <structmember.h>

#include "../kernels/ops.h" 
#include "../core/memory.h"

typedef struct {
    PyObject_HEAD
    void* d_ptr;
    int size;
} _Tensor;

static void _Tensor_dealloc(_Tensor* self) {
    if (self->d_ptr) {
        free_gpu(self->d_ptr);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int _Tensor_init(_Tensor* self, PyObject* args, PyObject* kwds) {
    int size = 0;
    if (!PyArg_ParseTuple(args, "i", &size)) {
        return -1;
    }
    
    self->size = size;
    self->d_ptr = alloc_gpu(size * sizeof(float));
    
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
        PyErr_SetString(PyExc_ValueError, "Size mismatch");
        return NULL;
    }

    _Tensor* result = (_Tensor*)Py_TYPE(self)->tp_alloc(Py_TYPE(self), 0);
    if (!result) return NULL;
    
    result->size = self->size;
    result->d_ptr = alloc_gpu(self->size * sizeof(float));

    launch_add(self->d_ptr, other->d_ptr, result->d_ptr, self->size);

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

    copy_to_gpu(self->d_ptr, temp_host, self->size * sizeof(float));
    free(temp_host);
    
    Py_RETURN_NONE;
}

static PyObject* _Tensor_to_list(_Tensor* self, PyObject* Py_UNUSED(ignored)) {
    float* temp_host = (float*)malloc(self->size * sizeof(float));
    copy_to_cpu(temp_host, self->d_ptr, self->size * sizeof(float));

    PyObject* list = PyList_New(self->size);
    for (int i=0; i<self->size; i++) {
        PyList_SetItem(list, i, PyFloat_FromDouble(temp_host[i]));
    }
    free(temp_host);
    return list;
}

static PyMemberDef _Tensor_members[] = {
    {"size", T_INT, offsetof(_Tensor, size), READONLY, "Size of the tensor"},
    {NULL}
};

static PyMethodDef _Tensor_methods[] = {
    {"_add", (PyCFunction)_Tensor_add, METH_VARARGS, "Low level add"},
    {"copy_from_list", (PyCFunction)_Tensor_copy_from_list, METH_VARARGS, "Load data"},
    {"to_list", (PyCFunction)_Tensor_to_list, METH_NOARGS, "Read data"},
    {NULL}
};

static PyTypeObject _TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ember._core._Tensor",
    .tp_basicsize = sizeof(_Tensor),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_members = _Tensor_members,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)_Tensor_init,
    .tp_dealloc = (destructor)_Tensor_dealloc,
    .tp_methods = _Tensor_methods,
};

static PyModuleDef core_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "ember._core",
    .m_doc = "Ember Core Backend",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__core(void) {
    PyObject* m;
    if (PyType_Ready(&_TensorType) < 0) return NULL;
    
    m = PyModule_Create(&core_module);
    if (m == NULL) return NULL;

    Py_INCREF(&_TensorType);
    PyModule_AddObject(m, "_Tensor", (PyObject*)&_TensorType);
    return m;
}