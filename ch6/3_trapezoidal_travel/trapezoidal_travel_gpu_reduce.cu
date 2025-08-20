#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>

#define N 10'000'000         // Increased number of speed measurements for GPU efficiency
#define T 0.25                // Time interval between measurements (15 minutes)

__global__ void trapezoidalKernel(double *speeds, double *result, int n) {
    extern __shared__ double sharedData[];
    int tid = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    double localDistance = 0.0;
    if (i < n - 1) {
        localDistance = 0.5 * (speeds[i] + speeds[i + 1]) * T;
    }
    sharedData[tid] = localDistance;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

double trapezoidalCpu(double *speeds, int n) {
    double totalDistance = 0.0;
    for (int i = 0; i < n - 1; i++) {
        totalDistance += 0.5 * (speeds[i] + speeds[i + 1]) * T;
    }
    return totalDistance;
}

void generateRandomSpeeds(double *speeds, int n) {
    for (int i = 0; i < n; i++) {
        speeds[i] = 40.0 + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (80.0 - 40.0)));
    }
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    double *h_speeds = (double*)malloc(N * sizeof(double));
    generateRandomSpeeds(h_speeds, N);

    auto startCPU = std::chrono::high_resolution_clock::now();
    double cpuResult = trapezoidalCpu(h_speeds, N);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endCPU - startCPU;
    std::cout << "CPU Distance: " << cpuResult << " km" << std::endl;
    std::cout << "CPU Execution Time: " << cpuDuration.count() << " ms" << std::endl;

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    double *d_speeds, *d_partialResults;
    cudaMalloc(&d_speeds, N * sizeof(double));
    cudaMalloc(&d_partialResults, gridSize * sizeof(double));

    cudaMemcpy(d_speeds, h_speeds, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t startGPU, endGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&endGPU);
    cudaEventRecord(startGPU);

    trapezoidalKernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_speeds, d_partialResults, N);
    cudaDeviceSynchronize();

    double *h_partialResults = (double*)malloc(gridSize * sizeof(double));
    cudaMemcpy(h_partialResults, d_partialResults, gridSize * sizeof(double), cudaMemcpyDeviceToHost);
    
    double gpuResult = 0.0;
    for (int i = 0; i < gridSize; i++) {
        gpuResult += h_partialResults[i];
    }

    cudaEventRecord(endGPU);
    cudaEventSynchronize(endGPU);

    float gpuDuration = 0.0;
    cudaEventElapsedTime(&gpuDuration, startGPU, endGPU);

    std::cout << "GPU Distance: " << gpuResult << " km" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " ms" << std::endl;


    free(h_speeds);
    free(h_partialResults);
    cudaFree(d_speeds);
    cudaFree(d_partialResults);
    cudaEventDestroy(startGPU);
    cudaEventDestroy(endGPU);

    return 0;
}