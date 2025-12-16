#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <structmember.h>
#include "../kernels/ops.h"
#include "../core/memory.h"

// shadow class that stores contigous tensor
typedef struct {
    PyObject_HEAD
    void* d_ptr;
    int size;
} CTensor;

// deallocator, called when object is collected
static void CTensor_dealloc(CTensor* self) {
    if (self->d_ptr) {
        free_gpu(self->d_ptr);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// allocate memeory for class
static int CTensor_init(CTensor* self, PyObject* args, PyObject* kwds) {
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

static PyObject* CTensor_add(CTensor* self, PyObject* args) {
    PyObject* other_obj;
    if (!PyArg_ParseTuple(args, "O", &other_obj)) return NULL;

    // cast
    CTensor* other = (CTensor*)other_obj;

    // backend check
    if (self->size != other->size) {
        PyErr_SetString(PyExc_ValueError, "Size mismatch in backend");
        return NULL;
    }

    // return object 
    CTensor* result = (CTensor*)PyObject_CallObject((PyObject*)Py_TYPE(self), NULL);
    
    result->size = self->size;
    result->d_ptr = (float*)alloc_gpu(self->size * sizeof(float));

    // call cuda
    launch_add(self->d_ptr, other->d_ptr, result->d_ptr, self->size);

    return (PyObject*)result;
}

static PyObject* CTensor_copy_from_list(CTensor* self, PyObject* args) {
    PyObject* py_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &py_list)) return NULL;

    // In real code, add checks here.
    float* temp_host = (float*)malloc(self->size * sizeof(float));
    
    for (int i=0; i<self->size; i++) {
        PyObject* item = PyList_GetItem(py_list, i);
        temp_host[i] = (float)PyFloat_AsDouble(item);
    }

    copy_to_gpu(self->d_ptr, temp_host, self->size * sizeof(float));
    free(temp_host);
    
    Py_RETURN_NONE;
}

static PyObject* CTensor_to_list(CTensor* self, PyObject* Py_UNUSED(ignored)) {
    float* temp_host = (float*)malloc(self->size * sizeof(float));
    copy_to_cpu(temp_host, self->d_ptr, self->size * sizeof(float));

    PyObject* list = PyList_New(self->size);
    for (int i=0; i<self->size; i++) {
        PyList_SetItem(list, i, PyFloat_FromDouble(temp_host[i]));
    }
    free(temp_host);
    return list;
}

static PyMethodDef CTensor_methods[] = {
    {"_add", (PyCFunction)CTensor_add, METH_VARARGS, "Low level add"},
    {"copy_from_list", (PyCFunction)CTensor_copy_from_list, METH_VARARGS, "Load data"},
    {"to_list", (PyCFunction)CTensor_to_list, METH_NOARGS, "Read data"},
    {NULL}
};

static PyTypeObject CTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_core._CTensor",
    .tp_basicsize = sizeof(CTensor),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)CTensor_init,
    .tp_dealloc = (destructor)CTensor_dealloc,
    .tp_methods = CTensor_methods,
};

static PyModuleDef core_module = {
    PyModuleDef_HEAD_INIT, .m_name = "_core", .m_size = -1,
};

PyMODINIT_FUNC PyInit_my_engine_core(void) {
    PyObject* m;
    if (PyType_Ready(&CTensorType) < 0) return NULL;
    m = PyModule_Create(&core_module);
    Py_INCREF(&CTensorType);
    PyModule_AddObject(m, "_CTensor", (PyObject*)&CTensorType);
    return m;
}