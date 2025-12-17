#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <random>

#define N 7000

__global__ void matrixAddKernel(const double* A, const double* B, double* C, int width) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < width && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

void matrixAddCpu(const double* A, const double* B, double* C, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            C[i * width + j] = A[i * width + j] + B[i * width + j];
        }
    }
}

void initializeMatrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

bool checkResults(double *A, double *B, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(A[i] - B[i]) > 1e-10) {
            return false;
        }
    }
    return true;
}



int main() {
    
    srand(static_cast<unsigned int>(time(0)));

    int matrixSize = N * N * sizeof(double);

    double *h_A = (double*)malloc(matrixSize);
    double *h_B = (double*)malloc(matrixSize);
    double *h_C_CPU = (double*)malloc(matrixSize);
    double *h_C_GPU = (double*)malloc(matrixSize);


    initializeMatrix(h_A, N * N);
    initializeMatrix(h_B, N * N);

    double *d_A;
    double *d_B; 
    double *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);

    auto startTime = std::chrono::high_resolution_clock::now();
    matrixAddCpu(h_A, h_B, h_C_CPU, N);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;

    std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl;

    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuDuration;
    cudaEventElapsedTime(&gpuDuration, start, stop);

    std::cout << "Time taken by GPU: " << gpuDuration << " ms\n";

    cudaMemcpy(h_C_GPU, d_C, matrixSize, cudaMemcpyDeviceToHost);

    if (checkResults(h_C_CPU, h_C_GPU, N * N)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
