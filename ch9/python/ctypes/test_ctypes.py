import ctypes
import numpy as np

lib = ctypes.CDLL("../../build/libvector_add.so")

lib.vectorAdd.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int
]

N = 10000000
a = np.array(range(N), dtype=np.int32)
b = np.array(range(N, 2*N), dtype=np.int32)
c = np.zeros(N, dtype=np.int32)

lib.vectorAdd(
    a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    N
)

print("Result:", c)
