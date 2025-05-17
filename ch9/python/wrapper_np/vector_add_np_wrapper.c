#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*vectorAddFunc)(int *, int *, int *, int);

static vectorAddFunc vectorAdd = NULL;

static PyObject* pyVectorAdd(PyObject* self, PyObject* args) {
    PyArrayObject *a, *b, *c;
    int N;

    if (!PyArg_ParseTuple(args, "OOOi",
                           &a,
                           &b,
                           &c,
                          &N)) {
        return NULL;
    }

    int *a_ptr = (int*)PyArray_DATA(a);
    int *b_ptr = (int*)PyArray_DATA(b);
    int *c_ptr = (int*)PyArray_DATA(c);

    vectorAdd(a_ptr, b_ptr, c_ptr, N);

    Py_RETURN_NONE;
}

static PyMethodDef VectorAddMethods[] = {
    {"vectorAdd", pyVectorAdd, METH_VARARGS, "Perform vector addition using CUDA"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef vectoraddmodule = {
    PyModuleDef_HEAD_INIT,
    "vector_add_np_wrapper",
    NULL,
    -1,
    VectorAddMethods
};

PyMODINIT_FUNC PyInit_vector_add_np_wrapper(void) {
    void *handle = dlopen("../../build/libvector_add.so", RTLD_LAZY);
    if (!handle) {
        PyErr_SetString(PyExc_ImportError, "Could not load libvector_add.so");
        return NULL;
    }

    vectorAdd = (vectorAddFunc)dlsym(handle, "vectorAdd");
    if (!vectorAdd) {
        PyErr_SetString(PyExc_ImportError, "Could not find vectorAdd in libvector_add.so");
        return NULL;
    }

    import_array();

    return PyModule_Create(&vectoraddmodule);
}
