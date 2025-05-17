#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*vectorAddFunc)(int *, int *, int *, int);

static vectorAddFunc vectorAdd = NULL;

static PyObject* pyVectorAdd(PyObject* self, PyObject* args) {
    PyObject *py_a, *py_b;
    int N;

    if (!PyArg_ParseTuple(args, "OOi", &py_a, &py_b, &N))
        return NULL;

    if (!PyList_Check(py_a) || !PyList_Check(py_b) || PyList_Size(py_a) != N || PyList_Size(py_b) != N) {
        PyErr_SetString(PyExc_ValueError, "Arguments must be lists of same size N.");
        return NULL;
    }

    int *a = (int*)malloc(N * sizeof(int));
    int *b = (int*)malloc(N * sizeof(int));
    int *c = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = (int)PyLong_AsLong(PyList_GetItem(py_a, i));
        b[i] = (int)PyLong_AsLong(PyList_GetItem(py_b, i));
    }

    vectorAdd(a, b, c, N);

    PyObject* result = PyList_New(N);
    for (int i = 0; i < N; i++) {
        PyList_SetItem(result, i, PyLong_FromLong(c[i]));
    }

    free(a);
    free(b);
    free(c);

    return result;
}

static PyMethodDef VectorAddMethods[] = {
    {"vectorAdd", pyVectorAdd, METH_VARARGS, "Perform vector addition using CUDA"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef vectoraddmodule = {
    PyModuleDef_HEAD_INIT,
    "vector_add_wrapper",
    NULL,
    -1,
    VectorAddMethods
};

PyMODINIT_FUNC PyInit_vector_add_wrapper(void) {
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

    return PyModule_Create(&vectoraddmodule);
}
