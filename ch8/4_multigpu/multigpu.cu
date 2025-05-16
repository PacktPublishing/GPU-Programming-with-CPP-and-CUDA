#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void vectorMatrixMulKernel(float* d_vec, float* d_mat, float* d_res, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += d_mat[row * cols + col] * d_vec[col];
        }
        d_res[row] = sum;
    }
}

void vectorMatrixMulCpu(float* vec, float* mat, float* res, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += mat[row * cols + col] * vec[col];
        }
        res[row] = sum;
    }
}

void compareResults(float* cpuRes, float* gpuRes, int rows) {
    for (int i = 0; i < rows; i++) {
        if (fabs(cpuRes[i] - gpuRes[i]) > 1e-4) {
            std::cerr << "Mismatch at " << i << " CPU: " << cpuRes[i] << " GPU: " << gpuRes[i] << " diff=" << abs(cpuRes[i] - gpuRes[i]) << std::endl;
            return;
        }
    }
    std::cout << "Results match!" << std::endl;
}

void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}


int main() {
    srand(static_cast<unsigned int>(time(0)));

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    int cols = 16380;
    int rows = cols;
    int chunk_size = rows / deviceCount;

    float *h_vec = (float*)malloc(cols * sizeof(float));
    float *h_res_cpu = (float*)malloc(cols * sizeof(float));
    float *h_mat = (float*)malloc(rows * cols * sizeof(float));
    float *h_res_gpu = (float*)malloc(rows * sizeof(float));


    initializeMatrix(h_vec, cols);
    initializeMatrix(h_mat, rows * cols);

    float *d_vec[deviceCount];
    float *d_mat[deviceCount];
    float *d_res[deviceCount];
    for (int device = 0; device < deviceCount; device++) {
        cudaSetDevice(device);
        cudaMalloc(&d_vec[device], cols * sizeof(float));
        cudaMalloc(&d_mat[device], chunk_size * cols * sizeof(float));
        cudaMalloc(&d_res[device], chunk_size * sizeof(float));

        cudaMemcpy(d_vec[device], h_vec, cols * sizeof(float), cudaMemcpyHostToDevice);
    }

    auto gpuStart = std::chrono::high_resolution_clock::now();


    for (int device = 0; device < deviceCount; device++) {
        cudaSetDevice(device);
        cudaMemcpy(d_mat[device], h_mat + device * chunk_size * cols, chunk_size * cols * sizeof(float), cudaMemcpyHostToDevice);
    }

    for (int device = 0; device < deviceCount; device++) {
        cudaSetDevice(device);
        int blocks = (chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        vectorMatrixMulKernel<<<blocks, BLOCK_SIZE>>>(d_vec[device], d_mat[device], d_res[device], chunk_size, cols);
    }

    for (int device = 0; device < deviceCount; device++) {
        cudaSetDevice(device);

        cudaMemcpy(h_res_gpu + device * chunk_size, d_res[device], chunk_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_vec[device]);
        cudaFree(d_mat[device]);
        cudaFree(d_res[device]);
    }

    auto gpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> gpuDuration = gpuEnd - gpuStart;
    std::cout << "Time taken by GPU: " << gpuDuration.count() << " ms" << std::endl;

    auto cpuStart = std::chrono::high_resolution_clock::now();
    vectorMatrixMulCpu(h_vec, h_mat, h_res_cpu, rows, cols);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpuDuration = cpuEnd - cpuStart;
    std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl;

    compareResults(h_res_cpu, h_res_gpu, rows);

    return 0;
}
