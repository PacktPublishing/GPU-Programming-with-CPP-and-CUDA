#include <cuda_runtime.h>
#include <iostream>

#define SIZE_MB(x) (long(x) * 1024 * 1024)  // Convert MB to bytes

void measureMemcpyBandwidth(long dataSize) {
    float *h_data, *d_data;
    cudaMallocHost(&h_data, dataSize);
    cudaMalloc(&d_data, dataSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpyAsync(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float bandwidth = (dataSize / (milliseconds / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Data Size: " << dataSize / (1024 * 1024) << " MB, Time: " << milliseconds << " ms, Bandwidth: " << bandwidth << " GB/s\n";

    cudaFreeHost(h_data);
    cudaFree(d_data);
}

int main() {
    std::cout << "Testing different transfer sizes:\n";
    measureMemcpyBandwidth(SIZE_MB(1));
    measureMemcpyBandwidth(SIZE_MB(10));
    measureMemcpyBandwidth(SIZE_MB(20));
    measureMemcpyBandwidth(SIZE_MB(30));
    measureMemcpyBandwidth(SIZE_MB(32));
    measureMemcpyBandwidth(SIZE_MB(40));
    measureMemcpyBandwidth(SIZE_MB(50));
    measureMemcpyBandwidth(SIZE_MB(100));
    measureMemcpyBandwidth(SIZE_MB(200));
    measureMemcpyBandwidth(SIZE_MB(300));
    measureMemcpyBandwidth(SIZE_MB(400));
    measureMemcpyBandwidth(SIZE_MB(500));
    measureMemcpyBandwidth(SIZE_MB(600));
    measureMemcpyBandwidth(SIZE_MB(700));
    measureMemcpyBandwidth(SIZE_MB(800));
    measureMemcpyBandwidth(SIZE_MB(900));
    measureMemcpyBandwidth(SIZE_MB(1024)); // 1GB

    return 0;
}
