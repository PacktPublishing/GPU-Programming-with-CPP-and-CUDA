#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <chrono>

#define N 2000

__global__ void matrixMulKernel(double *A, double *B, double *C, int width) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < width && col < width) {
        double value = 0;
        for (int k = 0; k < width; k++) {
            value += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = value;
    }
}

void matrixMulCpu(double *A, double *B, double *C, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            double value = 0;
            for (int k = 0; k < width; k++) {
                value += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = value;
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

    auto startCpu = std::chrono::high_resolution_clock::now();
    matrixMulCpu(h_A, h_B, h_C_CPU, N);
    auto stopCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = (stopCpu - startCpu);

    std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl;

    double *d_A;
    double *d_B;
    double *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);

    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTimeToCopy = 0;
    cudaEventElapsedTime(&gpuTimeToCopy, start, stop);
    std::cout << "GPU memory copy time: " << gpuTimeToCopy << " ms" << std::endl;

    cudaEventRecord(start);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float gpuDuration;
    cudaEventElapsedTime(&gpuDuration, start, stop);
    std::cout << "GPU execution time: " << gpuDuration << " ms" << std::endl;
    

    cudaEventRecord(start);

    cudaMemcpy(h_C_GPU, d_C, matrixSize, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float gpuTimeToRetrieve = 0;
    cudaEventElapsedTime(&gpuTimeToRetrieve, start, stop);
    std::cout << "GPU memory retrieve time: " << gpuTimeToRetrieve << " ms" << std::endl;
    std::cout << "GPU total time: " << (gpuDuration + gpuTimeToCopy + gpuTimeToRetrieve) << " ms" << std::endl;

    double cpuTotalTime = cpuDuration.count();
    double gpuTotalTime = (gpuDuration + gpuTimeToCopy + gpuTimeToRetrieve);
    std::cout << "speed up: " << cpuTotalTime / gpuTotalTime << std::endl;

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