import vector_add_wrapper as vaw
import random

N = 10000000
a = [random.randint(0, N) for _ in range(N)]
b = [random.randint(N, 2*N) for _ in range(N)]

result = vaw.vectorAdd(a, b, N)
print("Result:", result)
