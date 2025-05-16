#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>

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

void vectorMatrixMulCpu(float *vec, float *mat, float *res, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        res[i] = 0;
        for (int j = 0; j < cols; j++) {
            res[i] += mat[i * cols + j] * vec[j];
        }
    }
}

void checkResult(float *cpuRes, float *gpuRes, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpuRes[i] - gpuRes[i]) > 1e-3) {
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

void compute(int cols) {
    int rows = cols;

    float *h_vec = (float*)malloc(cols * sizeof(float));
    float *h_mat = (float*)malloc(rows * cols * sizeof(float));
    initializeMatrix(h_vec, cols);
    initializeMatrix(h_mat, rows * cols);

    float *h_res_gpu = (float*)malloc(rows * sizeof(float));
    float *h_res_cpu = (float*)malloc(rows * sizeof(float));

    float *d_vec;
    float *d_mat;
    float *d_res;
    cudaMalloc(&d_vec, cols * sizeof(float));
    cudaMalloc(&d_mat, rows * cols * sizeof(float));
    cudaMalloc(&d_res, rows * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpy(d_vec, h_vec, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, h_mat, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorMatrixMulKernel<<<blocks, BLOCK_SIZE>>>(d_vec, d_mat, d_res, rows, cols);
    cudaMemcpy(h_res_gpu, d_res, rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, start, stop);
    std::cout << "Time taken by GPU: " << gpuDuration << " ms, for matrix: " << cols << "x" << cols << std::endl;


    auto startCpu = std::chrono::high_resolution_clock::now();
    vectorMatrixMulCpu(h_vec, h_mat, h_res_cpu, rows, cols);
    auto stopCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = (stopCpu - startCpu);

    std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl;

    
    checkResult(h_res_cpu, h_res_gpu, rows);

    cudaFree(d_vec);
    cudaFree(d_mat);
    cudaFree(d_res);

}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    compute(16380);
    compute(32760);

    return 0;
}
