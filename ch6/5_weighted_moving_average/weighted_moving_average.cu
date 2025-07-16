#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <chrono>

#define NUM_SENSORS 50'000'000
#define NUM_READINGS 3

void rotateIndices(int *indices) {
    int newData = indices[0];
    indices[0] = indices[1];
    indices[1] = indices[2];
    indices[2] = newData;
}

__global__ void smoothSensorsKernel(float *buffers, int *indices, float *output, float *weights) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NUM_SENSORS) return;

    float val = 0.0f;
    for (int i=0; i < NUM_READINGS; i++){
        val = weights[i] * buffers[indices[i] * NUM_READINGS + idx];
    }

    output[idx] = val;
}

void smoothSensorsCpu(float *buffers, int *indices, float *output, float *weights){
    for (int i = 0; i < NUM_READINGS; i++) {
        for (int j = 0; j < NUM_SENSORS; j++){
            output[j] = weights[i] * buffers[indices[i] * NUM_READINGS + j];
        }
    }
}

bool checkResults(float *A, float *B, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(A[i] - B[i]) > 1e-10) {
            std::cerr << "Mismatch at index " << i << 
            " CPU: " << A[i] << 
            " GPU: " << B[i] << 
            " diff=" << abs(A[i] - B[i]) << std::endl;
            return false;
        }
    }
    return true;
}

void initializeBuffer(float *h_buffers, int bufferId)
{
    for (int j = 0; j < NUM_SENSORS; j++) {
        h_buffers[bufferId + j] = 1.0f + ((rand() % 100) / 1000.0f);
    }
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    int bufferSize = NUM_SENSORS * sizeof(float);

    float  h_weights[NUM_READINGS] = {0.2f, 0.3f, 0.5f};
    int    h_indices[NUM_READINGS] = {0, 1, 2};
    float *h_buffers = (float*) malloc(NUM_READINGS * bufferSize);
    float *h_output_GPU = (float*) malloc(bufferSize);
    float *h_output_CPU = (float*) malloc(bufferSize);

    float *d_buffers;
    float *d_output;
    float *d_weights;
    int   *d_indices;

    std::chrono::duration<double, std::milli> cpuGlobalTime;
    int threads = 256;
    int blocks = (NUM_SENSORS + threads - 1) / threads;
    float gpuGlobalTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_buffers, NUM_READINGS * bufferSize);
    cudaMalloc(&d_output, bufferSize);
    cudaMalloc(&d_indices, NUM_READINGS * sizeof(int));
    cudaMalloc(&d_weights, NUM_READINGS * sizeof(float));

    cudaMemcpy(d_weights, h_weights, NUM_READINGS * sizeof(float), cudaMemcpyHostToDevice);
    
    for(int i = 0; i < 4; i++) {

        std::cout << i << std::endl;

        int newDataIdx = h_indices[0];

        if (i == 0) {
            for(int j = 0; j < NUM_READINGS; j++) {
                initializeBuffer(h_buffers, j);
            }
        } else {
            newDataIdx = h_indices[0];
            rotateIndices(h_indices);
            initializeBuffer(h_buffers, newDataIdx); //one new reading for all sensors
        }
        
        auto startCpu = std::chrono::high_resolution_clock::now();
        smoothSensorsCpu(h_buffers, h_indices, h_output_CPU, h_weights);
        auto stopCpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpuDuration = (stopCpu - startCpu);
        cpuGlobalTime += cpuDuration;
        std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl;


        cudaEventRecord(start);

        cudaMemcpy(d_indices, h_indices, NUM_READINGS * sizeof(int), cudaMemcpyHostToDevice);
        if (i == 0) {
            cudaMemcpy(d_buffers, h_buffers, NUM_READINGS * bufferSize, cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(&d_buffers[newDataIdx], &h_buffers[newDataIdx], bufferSize, cudaMemcpyHostToDevice);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpuTimeToCopy = 0;
        cudaEventElapsedTime(&gpuTimeToCopy, start, stop);
        std::cout << "GPU memory copy time: " << gpuTimeToCopy << " ms" << std::endl;

        cudaEventRecord(start);
        smoothSensorsKernel<<<blocks, threads>>>(d_buffers, d_indices, d_output, d_weights);
        
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float gpuDuration = 0;
        cudaEventElapsedTime(&gpuDuration, start, stop);
        std::cout << "GPU execution time: " << gpuDuration << " ms" << std::endl;
    
        cudaEventRecord(start);
        cudaMemcpy(h_output_GPU, d_output, bufferSize, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float gpuTimeToRetrieve = 0;
        cudaEventElapsedTime(&gpuTimeToRetrieve, start, stop);
        std::cout << "GPU memory retrieve time: " << gpuTimeToRetrieve << " ms" << std::endl;

        float gpuTotalTime = (gpuDuration + gpuTimeToCopy + gpuTimeToRetrieve);
        gpuGlobalTime += gpuTotalTime;
        std::cout << "GPU total time: " << gpuTotalTime << " ms" << std::endl;

        if (!checkResults(h_output_CPU, h_output_GPU, NUM_SENSORS)) {
            std::cout << "Results do not match!" << std::endl;
        }
    }

    std::cout << "=============================================" << std::endl;
    std::cout << "CPU global time: " << cpuGlobalTime.count() << " ms" << std::endl;
    std::cout << "GPU global time: " << gpuGlobalTime << " ms" << std::endl;

    std::cout << "speed up over all executions: " << cpuGlobalTime.count() / gpuGlobalTime << std::endl;

    cudaFree(d_buffers);
    cudaFree(d_output);
    cudaFree(d_indices);
    cudaFree(d_weights);
    free(h_output_CPU);
    free(h_output_GPU);
    free(h_buffers);

    return 0;
}
