import numpy as np
from vector_add import vectorAdd

def testVectorAdd():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    expected = a + b

    result = vectorAdd(a, b)

    for i in range(len(expected)):
        assert abs(result[i] - expected[i]) < 1e-5, f"Mismatch at index {i}: got {result[i]}, expected {expected[i]}"

    np.testing.assert_allclose(result, expected, rtol=1e-5)