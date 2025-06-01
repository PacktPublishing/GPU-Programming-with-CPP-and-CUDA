#include <gtest/gtest.h>
#include "../include/vector_add.h"

TEST(VectorAddTest, SimpleAddition) {
    int N = 5;
    float A[N] = {1, 2, 3, 4, 5};
    float B[N] = {10, 20, 30, 40, 50};
    float C[N] = {0};

    vectorAdd(A, B, C, N);

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(C[i], A[i] + B[i]);
    }
}