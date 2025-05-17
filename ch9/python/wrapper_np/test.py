import vector_add_np_wrapper as vaw_np
import numpy as np
import time
import argparse

def calculate(size):
    N = int (size)
    a = np.random.randint(0, N, size=N, dtype=np.int32)
    b = np.random.randint(N, 2*N, size=N, dtype=np.int32)
    c = np.zeros(N, dtype=np.int32)

    startTime = time.time()
    vaw_np.vectorAdd(a, b, c, N)
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