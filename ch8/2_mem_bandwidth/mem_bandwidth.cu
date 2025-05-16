#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int memClockMHz = prop.memoryClockRate / 1000;  // Convert to MHz
    int busWidth = prop.memoryBusWidth;  // In bits
    float bandwidthInBytes = memClockMHz * busWidth / 8; // In bytes
    float bandwidthInGB = bandwidthInBytes / 1024; // In GigaBytes
    float bandwidthGBs = 2.0f * bandwidthInGB; // DDR memory is twice the clock rate

    int device;
    cudaGetDevice(&device);

    int concurrentKernels = 0;
    int asyncEngineCount = 0;

    cudaDeviceGetAttribute(&concurrentKernels, cudaDevAttrConcurrentKernels, device);
    cudaDeviceGetAttribute(&asyncEngineCount, cudaDevAttrAsyncEngineCount, device);

    std::cout << "Device " << device << ":" << std::endl;
    std::cout << "  Concurrent Kernel Execution: " << (concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "  Async Engine Count: " << asyncEngineCount << std::endl;

    std::cout << "Memory Clock Speed: " << memClockMHz << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << busWidth << " bits" << std::endl;
    std::cout << "Estimated Memory Bandwidth: " << bandwidthGBs << " GB/s" << std::endl;

    return 0;
}
