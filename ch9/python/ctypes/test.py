import ctypes
import numpy as np
import time
import argparse

lib = ctypes.CDLL("../../build/libvector_add.so")

lib.vectorAdd.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int
]

def calculate(size):
    N = int (size)
    a = np.array(range(N), dtype=np.int32)
    b = np.array(range(N, 2*N), dtype=np.int32)
    c = np.zeros(N, dtype=np.int32)

    startTime = time.time()
    lib.vectorAdd(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        N
    )
    endTime = time.time()

    executionTimeInMs = (endTime - startTime) * 1000

    print(f"Function execution time: {executionTimeInMs:.2f} ms")

def main():
    parser = argparse.ArgumentParser(description="N")
    parser.add_argument("N", help="The size of the array")

    args = parser.parse_args()

    calculate(args.N)

if __name__ == "__main__":
    main()