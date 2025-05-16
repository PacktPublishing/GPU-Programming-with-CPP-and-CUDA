#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cfloat>
#include <iomanip>
#include <cuda_runtime.h>

__global__ void oddEvenSortStepKernel(double *arr, int size, bool *swapped, bool isOddPhase) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i = isOddPhase ? 2 * idx + 1 : 2 * idx;

    if (i < size - 1) {
        if (arr[i] > arr[i + 1]) {
            double temp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = temp;
            *swapped = true;
        }
    }
}

void oddEvenSortGpu(double *arr, int size) {
    double *d_arr;
    bool *d_swapped;
    bool h_swapped;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    cudaMalloc((void **)&d_arr, size * sizeof(double));
    cudaMalloc((void **)&d_swapped, sizeof(bool));

    cudaMemcpy(d_arr, arr, size * sizeof(double), cudaMemcpyHostToDevice);

    do {
        h_swapped = false;
        cudaMemcpy(d_swapped, &h_swapped, sizeof(bool), cudaMemcpyHostToDevice);

        // Odd phase
        oddEvenSortStepKernel<<<blocks, threads>>>(d_arr, size, d_swapped, true);
        cudaDeviceSynchronize();

        // Even phase
        oddEvenSortStepKernel<<<blocks, threads>>>(d_arr, size, d_swapped, false);
        cudaDeviceSynchronize();

        // Check if any swaps occurred
        cudaMemcpy(&h_swapped, d_swapped, sizeof(bool), cudaMemcpyDeviceToHost);

    } while (h_swapped);

    cudaMemcpy(arr, d_arr, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_swapped);
}

void oddEvenSortCpu(double *arr, int size) {
    bool swapped;

    do {
        swapped = false;

        // Odd phase
        for (int i = 1; i < size - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                std::swap(arr[i], arr[i + 1]);
                swapped = true;
            }
        }

        // Even phase
        for (int i = 0; i < size - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                std::swap(arr[i], arr[i + 1]);
                swapped = true;
            }
        }
    } while (swapped);
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    int n = 100'000;
    double *h_data = (double*)malloc(n * sizeof(double));
    double *h_data_gpu = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        h_data[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    double* d_data;
    cudaMalloc(&d_data, n * sizeof(double));
    cudaMemcpy(d_data, h_data, n * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t startGpu, stopGpu;
    cudaEventCreate(&startGpu);
    cudaEventCreate(&stopGpu);

    cudaEventRecord(startGpu);

    oddEvenSortGpu(d_data, n);

    cudaEventRecord(stopGpu);

    cudaEventSynchronize(stopGpu);
    float gpuDuration;
    cudaEventElapsedTime(&gpuDuration, startGpu, stopGpu);
    std::cout << "GPU sorting time: " << gpuDuration << " ms" << std::endl;

    cudaMemcpy(h_data_gpu, d_data, n * sizeof(double), cudaMemcpyDeviceToHost);


    auto start = std::chrono::high_resolution_clock::now();
    oddEvenSortCpu(h_data, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = end - start;
    std::cout << "CPU sorting time: " << cpuDuration.count() << " ms" << std::endl;

    double max_difference = 0;
    for (int i = 0; i < n; i++) {
        max_difference = std::max(max_difference, std::abs(h_data[i] - h_data_gpu[i]));
    }
    std::cout << "Max difference between CPU and GPU results: " << max_difference << std::endl;


    free(h_data);
    free(h_data_gpu);
    cudaFree(d_data);
    cudaEventDestroy(startGpu);
    cudaEventDestroy(stopGpu);

    return 0;
}
