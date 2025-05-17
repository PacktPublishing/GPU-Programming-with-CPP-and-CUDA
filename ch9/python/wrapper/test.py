import vector_add_wrapper as vaw
import random
import time
import argparse

def calculate(size):
    N = int (size)

    a = [random.randint(0, N) for _ in range(N)]
    b = [random.randint(N, 2*N) for _ in range(N)]

    startTime = time.time()
    result = vaw.vectorAdd(a, b, N)
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