#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>

#define BLOCK_SIZE 256

__global__ void vectorMatrixMulKernel(float* d_vec, float* d_mat, float* d_res, int rows, int cols) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += d_mat[row * cols + col] * d_vec[col];
        }
        d_res[row] = sum;
    }
}

void vectorMatrixMulCpu(float* vec, float* mat, float* res, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        res[i] = 0;
        for (int j = 0; j < cols; j++) {
            res[i] += mat[i * cols + j] * vec[j];
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

void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

float compute(int chunkSize, int cols, bool computeCpuPart) {
    int rows = cols;
    float *h_vec = (float*)malloc(rows * sizeof(float));
    float *h_res_cpu = (float*)malloc(rows * sizeof(float));
    float *h_mat_pinned, *h_res_gpu;
    cudaMallocHost(&h_mat_pinned, rows * cols * sizeof(float));
    cudaMallocHost(&h_res_gpu, rows * sizeof(float));

    initializeMatrix(h_vec, cols);
    initializeMatrix(h_mat_pinned, rows * cols);

    float *d_vec;
    float *d_mat1;
    float *d_mat2;
    float *d_res1;
    float *d_res2;
    cudaMalloc(&d_vec, cols * sizeof(float));
    cudaMalloc(&d_mat1, chunkSize * cols * sizeof(float));
    cudaMalloc(&d_mat2, chunkSize * cols * sizeof(float));
    cudaMalloc(&d_res1, chunkSize * sizeof(float));
    cudaMalloc(&d_res2, chunkSize * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpy(d_vec, h_vec, cols * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int blocks = (chunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < rows; i += chunkSize * 2) {
        // Stream 1: Process first chunk
        cudaMemcpyAsync(d_mat1, h_mat_pinned + i * cols, chunkSize * cols * sizeof(float), cudaMemcpyHostToDevice, stream1);

        vectorMatrixMulKernel<<<blocks, BLOCK_SIZE, 0, stream1>>>(d_vec, d_mat1, d_res1, chunkSize, cols);

        cudaMemcpyAsync(h_res_gpu + i, d_res1, chunkSize * sizeof(float), cudaMemcpyDeviceToHost, stream1);

        // Stream 2: Process second chunk
        cudaMemcpyAsync(d_mat2, h_mat_pinned + (i + chunkSize) * cols, chunkSize * cols * sizeof(float), cudaMemcpyHostToDevice, stream2);

        vectorMatrixMulKernel<<<blocks, BLOCK_SIZE, 0, stream2>>>(d_vec, d_mat2, d_res2, chunkSize, cols);

        cudaMemcpyAsync(h_res_gpu + i + chunkSize, d_res2, chunkSize * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    }
    // Ensure first chunk has completed before reusing buffers
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float gpuDuration;
    cudaEventElapsedTime(&gpuDuration, start, stop);
    std::cout << "Time taken by GPU: " << gpuDuration << " ms, for matrix: " << cols << "x" << cols << " chunk size: " << chunkSize << std::endl;

    if (computeCpuPart) {
        auto startCpu = std::chrono::high_resolution_clock::now();
        vectorMatrixMulCpu(h_vec, h_mat_pinned, h_res_cpu, rows, cols);
        auto stopCpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpuDuration = (stopCpu - startCpu);

        std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl;

        checkResult(h_res_cpu, h_res_gpu, rows);
    }

    cudaFreeHost(h_mat_pinned);
    cudaFreeHost(h_res_gpu);
    cudaFree(d_vec);
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_res1);
    cudaFree(d_res2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return gpuDuration;
}

/*
* The average function will collect the gpuDuration for each chunkSize and matrix size
* over the iterations defined and calculate the percetage gain against the provided
* reference value of the execution time for the non streams application version.
*
* Note: you have to run the non streams version with the same matrix size.
*/
void average(int chunkSize, int cols, int iterations, float noStreamsReferenceTime) {
    long dataSize = (chunkSize * cols);
    long totalDataSize = (cols * cols);

    double sum = 0.0f;
    for(int i = 0; i < iterations; i++) {
        sum += compute(chunkSize, cols, false);
    }
    double average = sum / iterations;

    std::cout << " chunk: " << chunkSize
              << " data size: " << dataSize / 1024 / 1024 * 4 << "MB "
              << " total data size: " << totalDataSize / 1024 / 1024 * 4 << "MB "
              << " data partitions: " << (double)totalDataSize / (double)dataSize << " "
              << " average time: " << average << " ms"
              << " percentage gain: " << 100 - (average / noStreamsReferenceTime * 100) << "%"
              << std::endl;
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    compute(  273, 16380, true);
    compute(  546, 16380, true);
    compute(  819, 16380, true);
    compute( 1638, 16380, true);
    compute( 4095, 16380, true);
    compute( 8190, 16380, true);

    compute(  273, 32760, true);
    compute(  546, 32760, true);
    compute(  819, 32760, true);
    compute( 1638, 32760, true);
    compute( 4095, 32760, true);
    compute( 8190, 32760, true);
    compute(16380, 32760, true);

    /*
    * If you want to check an average of some executions against the values measured
    * for the no streams version you can use the following sample calls, but updating
    * the last value to the time measurement from your system.
    */
    // average(  273, 32760, 10, 500.615);
    // average(  546, 32760, 10, 500.615);
    // average(  819, 32760, 10, 500.615);
    // average( 1638, 32760, 10, 500.615);
    // average( 4095, 32760, 10, 500.615);
    // average( 8190, 32760, 10, 500.615);
    // average(16380, 32760, 10, 500.615);

    // average(  273, 16380, 10, 119.299);
    // average(  546, 16380, 10, 119.299);
    // average(  819, 16380, 10, 119.299);
    // average( 1638, 16380, 10, 119.299);
    // average( 4095, 16380, 10, 119.299);
    // average( 8190, 16380, 10, 119.299);

    return 0;
}