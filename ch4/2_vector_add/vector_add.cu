#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void vectorAddKernel(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void vectorAddCpu(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 100'000'000;
    size_t size = N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuCopyTime = 0;
    cudaEventElapsedTime(&gpuCopyTime, startEvent, stopEvent);

    std::cout<< std::fixed << "Time to copy data to GPU: " << gpuCopyTime << " ms" << std::endl;

    cudaEventRecord(startEvent, 0);

    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuExecutionTime = 0;
    cudaEventElapsedTime(&gpuExecutionTime, startEvent, stopEvent);

    std::cout<< std::fixed << "Time to execute on GPU: " << gpuExecutionTime << " ms" << std::endl;

    cudaEventRecord(startEvent, 0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuRetrieveTime = 0;
    cudaEventElapsedTime(&gpuRetrieveTime, startEvent, stopEvent);

    std::cout<< std::fixed << "Time taken to copy results back GPU: " << gpuRetrieveTime << " ms" << std::endl << std::endl;

    float gpuDuration = (gpuCopyTime + gpuExecutionTime + gpuRetrieveTime);
    std::cout << "Time taken by GPU: " << gpuDuration << " ms" << std::endl;


    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);


    auto start = std::chrono::high_resolution_clock::now();

    vectorAddCpu(h_A, h_B, h_C, N);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = (stop - start);

    std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl;
    std::cout << "========================================== " << std::endl;

    std::cout << "speed up (execution time only): " << cpuDuration.count() / gpuExecutionTime << std::endl;
    std::cout << "speed up (GPU total time): " << cpuDuration.count() / gpuDuration << std::endl;


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
