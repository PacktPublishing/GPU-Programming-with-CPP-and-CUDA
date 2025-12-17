#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void checkPrimeKernel(long long start, long long end) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long num = start + (tid * 2);
    bool isPrime = true;
    if (num <= 1) {
        isPrime = false;
        return;
    }
    if (num == 2) {
        isPrime = true;
        return;
    } 
    if (num % 2 == 0) {
        isPrime = false;
        return;
    }
    if (num > end) return;

    
    for (long long i = 3; i * i <= num; i += 2) {
        if (num % i == 0) {
            isPrime = false;
            break;
        }
    }

    /*
    * for study purposes we can print the verification of each number
    */
    //printf("tid=%d %lld is prime? %d\n", tid, num, isPrime);
}

bool checkPrimeCpu(long long num) {
    
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (long long i = 3; i * i <= num; i += 2) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}


int main() {
    long long start =  100'001LL; // must start with odd
    long long end   =  190'001LL;

    int threadsPerBlock = 256;
    int totalNumbers = (end - start) / 2 + 1;
    int blocksPerGrid = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    checkPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);

    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (long long num = start; num <= end; num += 2) {
        checkPrimeCpu(num);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;

    std::cout << "Time taken on CPU: " << std::fixed << cpuDuration.count() << " ms" << std::endl;
    std::cout << "speed up : " << cpuDuration.count() / gpuDuration << std::endl;

    return 0;
}
