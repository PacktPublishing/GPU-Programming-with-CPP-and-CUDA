#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <typeinfo>
#include <vector>

template <typename T>
void thrustSort(size_t size) {
    thrust::host_vector<T> h_vec(size);
    thrust::host_vector<T> h_result(size);
    
    std::cout << "Processing " << typeid(T).name() << " (" << size << " elements)" << std::endl;

    for (auto& val : h_vec) {
        val = static_cast<T>(rand()) / RAND_MAX;
    }

    auto cpuVec = h_vec;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    std::sort(cpuVec.begin(), cpuVec.end());
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = cpuEnd - cpuStart;
    std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl;


    auto gpuStart = std::chrono::high_resolution_clock::now();

    auto startCopy = std::chrono::high_resolution_clock::now();
    thrust::device_vector<T> d_vec = h_vec;
    auto endCopy = std::chrono::high_resolution_clock::now();

    auto startSort = std::chrono::high_resolution_clock::now();
    thrust::sort(d_vec.begin(), d_vec.end());
    auto endSort = std::chrono::high_resolution_clock::now();
    
    auto startCopyBack = std::chrono::high_resolution_clock::now();
    thrust::copy(d_vec.begin(), d_vec.end(), h_result.begin());
    auto endCopyBack = std::chrono::high_resolution_clock::now();

    auto gpuEnd = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> copyTime = endCopy - startCopy;
    std::chrono::duration<double, std::milli> copyBackTime = endCopyBack - startCopyBack;
    std::chrono::duration<double, std::milli> sortTime = endSort - startSort;
    std::chrono::duration<double, std::milli> gpuTotalTime = gpuEnd - gpuStart;

    std::cout << "GPU copy time: " << copyTime.count() << " ms" << std::endl;
    std::cout << "GPU copy back time: " << copyBackTime.count() << " ms" << std::endl;
    std::cout << "GPU sort time: " << sortTime.count() << " ms" << std::endl;
    std::cout << "Total time taken by GPU: " << gpuTotalTime.count() << " ms" << std::endl;
}

int main() {
    size_t N = 33'000'000;

    srand(static_cast<unsigned int>(time(0)));

    thrustSort<int>(N);
    std::cout << std::endl;
    thrustSort<double>(N);
    std::cout << std::endl;
    thrustSort<float>(N);

    return 0;
}
