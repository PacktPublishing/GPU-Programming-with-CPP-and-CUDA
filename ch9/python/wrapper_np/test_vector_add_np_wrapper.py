import vector_add_np_wrapper as vaw_np
import numpy as np

N = 10000000
a = np.random.randint(0, N, size=N, dtype=np.int32)
b = np.random.randint(N, 2*N, size=N, dtype=np.int32)
c = np.zeros(N, dtype=np.int32)

vaw_np.vectorAdd(a, b, c, N)
print("Result:", c)
