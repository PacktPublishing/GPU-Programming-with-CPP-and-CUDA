#include <iostream>
#include <cuda_runtime.h>
#include "../include/vector_add.h"

__global__ void vectorAddKernel(int *a, int *b, int *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {  
        c[idx] = a[idx] * b[idx];
    }
}

void vectorAdd(int *a, int *b, int *c, int N) {
    int *d_a;
    int *d_b;
    int *d_c;
    
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}