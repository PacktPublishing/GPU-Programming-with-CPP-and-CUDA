#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAddKernel(int *a, int *b, int *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {  
        c[idx] = a[idx] * b[idx];
    }

}

void vectorAddCpu(int *a, int *b, int *c, int N) {

    for (int idx = 0; idx < N; idx++) {  
        c[idx] = a[idx] + b[idx];
    }

}

void checkResult(int *cpuRes, int *gpuRes, int N) {
    for (int i = 0; i < N; ++i) {
        if (abs(cpuRes[i] - gpuRes[i]) > 1e-5) {
            std::cerr << "Mismatch at index " << i << 
            " CPU: " << cpuRes[i] << 
            " GPU: " << gpuRes[i] << 
            " diff=" << abs(cpuRes[i] - gpuRes[i]) << std::endl;
            return;
        }
    }
    std::cout << "Results match!" << std::endl;
}

int main() {
    const int N = 130;
    int h_a[N];
    int h_b[N];
    int h_c[N];
    int cpuResult[N];
    int *d_a;
    int *d_b;
    int *d_c;

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    vectorAddCpu(h_a, h_b, cpuResult, N);

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));


    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;

    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    checkResult(cpuResult, h_c, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
