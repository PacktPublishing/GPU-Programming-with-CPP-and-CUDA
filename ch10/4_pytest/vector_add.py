import ctypes
import numpy as np

lib = ctypes.CDLL('../3_gtest/build/libvector_add.so')

lib.vectorAdd.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

def vectorAdd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape
    n = a.size
    c = np.empty_like(a)
    lib.vectorAdd(a, b, c, n)
    return c
