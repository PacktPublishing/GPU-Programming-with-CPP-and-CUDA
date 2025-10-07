#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>

#define N 10'000'000           // Increased number of speed measurements for GPU efficiency
#define T 0.25                 // Time interval between measurements (15 minutes)

__global__ void trapezoidalKernel(double *speeds, double *distances, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n - 1) {
        distances[i] = 0.5 * (speeds[i] + speeds[i + 1]) * T;
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
    double *h_distances = (double*)malloc( (N - 1) * sizeof(double));

    generateRandomSpeeds(h_speeds, N);

    auto startCPU = std::chrono::high_resolution_clock::now();
    double cpuResult = trapezoidalCpu(h_speeds, N);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endCPU - startCPU;
    std::cout << "CPU Distance: " << cpuResult << " km" << std::endl;
    std::cout << "CPU Execution Time: " << cpuDuration.count() << " ms" << std::endl;

    double *d_speeds, *d_distances;
    cudaMalloc(&d_speeds, N * sizeof(double));
    cudaMalloc(&d_distances, (N - 1) * sizeof(double));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t startGPU, endGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&endGPU);
    cudaEventRecord(startGPU);

    cudaMemcpy(d_speeds, h_speeds, N * sizeof(double), cudaMemcpyHostToDevice);

    trapezoidalKernel<<<gridSize, blockSize>>>(d_speeds, d_distances, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_distances, d_distances, (N - 1) * sizeof(double), cudaMemcpyDeviceToHost);

    double gpuResult = 0.0;
    for (int i = 0; i < N - 1; i++) {
        gpuResult += h_distances[i];
    }

    cudaEventRecord(endGPU);
    cudaEventSynchronize(endGPU);
    float gpuDuration;
    cudaEventElapsedTime(&gpuDuration, startGPU, endGPU);

    std::cout << "GPU Distance: " << gpuResult << " km" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " ms" << std::endl;


    free(h_speeds);
    free(h_distances);
    cudaFree(d_speeds);
    cudaFree(d_distances);
    cudaEventDestroy(startGPU);
    cudaEventDestroy(endGPU);

    return 0;
}
