#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void matrixMultCpu(float* A, float* B, float* C, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

void checkResult(float* cpuRes, float* gpuRes, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpuRes[i] - gpuRes[i]) > 1e-3) {
            std::cerr << "Mismatch at " << i << " CPU: " << cpuRes[i] << " GPU: " << gpuRes[i] << " diff=" << abs(cpuRes[i] - gpuRes[i]) << std::endl;
            return;
        }
    }
    std::cout << "Results match!" << std::endl;
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    int N = 1024;
    size_t size = N * N * sizeof(float);

    float* h_A     = (float*)malloc(size);
    float* h_B     = (float*)malloc(size);
    float* h_C     = (float*)malloc(size);
    float* h_C_cpu = (float*)malloc(size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    auto startCpu = std::chrono::high_resolution_clock::now();
    matrixMultCpu(h_A, h_B, h_C_cpu, N);
    auto endCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endCpu - startCpu;
    std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl;

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    auto startTotal = std::chrono::high_resolution_clock::now();

    auto startCopy = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    auto endCopy = std::chrono::high_resolution_clock::now();

    auto startOverheadTime = std::chrono::high_resolution_clock::now();

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    auto endOverheadTime = std::chrono::high_resolution_clock::now();
    
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);
    

    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_T, // transpose both matrices
                N, N, N,
                &alpha,
                d_A, N,
                d_B, N,
                &beta,
                d_C, N);
    
    float* d_C_fixed;
    cudaMalloc(&d_C_fixed, size);

    cublasSgeam(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, N, // dimensions of the output matrix
                &alpha,
                d_C, N, // input matrix (to be transposed)
                &beta,
                nullptr, N, // second matrix not used
                d_C_fixed, N); // result goes here

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float gpuComputeTime = 0;
    cudaEventElapsedTime(&gpuComputeTime, startEvent, stopEvent);

    auto startCopyBack = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C, d_C_fixed, size, cudaMemcpyDeviceToHost);
    auto endCopyBack = std::chrono::high_resolution_clock::now();
    auto endTotal = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std:: milli> copyTime = endCopy - startCopy;
    std::chrono::duration<double, std:: milli> copyBackTime = endCopyBack - startCopyBack;
    std::chrono::duration<double, std:: milli> totalTime = endTotal - startTotal;
    std::chrono::duration<double, std:: milli> overheadTime = endOverheadTime - startOverheadTime;

    std::cout << "Device data copy time: " << copyTime.count() << " ms" << std::endl;
    std::cout << "cuBLAS overhead time: " << overheadTime.count() << " ms" << std::endl;
    std::cout << "Device compute time: " << gpuComputeTime << " ms" << std::endl;
    std::cout << "Device data copy back time: " << copyBackTime.count() << " ms" << std::endl;
    std::cout << "Total time taken by GPU: " << totalTime.count() << " ms" << std::endl;

    checkResult(h_C_cpu, h_C, N * N);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_fixed);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);

    return 0;
}
