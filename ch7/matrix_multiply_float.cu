#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define N 2000  // Define matrix size
#define TILE 16 // Define the tile size 

__global__ void matrixMulKernel(float *A, float *B, float *C, int width) {
    __shared__ float Asub[TILE][TILE];
    __shared__ float Bsub[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = ty + blockIdx.y * blockDim.y;
    int col = tx + blockIdx.x * blockDim.x;

    float sum = 0.0f;

    for (int i = 0; i < (width + blockDim.x - 1) / blockDim.x; i++) {
        if (row < width && (i * blockDim.x + tx) < width)
            Asub[ty][tx] = A[row * width + i * blockDim.x + tx];
        else
            Asub[ty][tx] = 0.0f;

        if (col < width && (i * blockDim.y + ty) < width)
            Bsub[ty][tx] = B[(i * blockDim.y + ty) * width + col];
        else
            Bsub[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < blockDim.x; k++) {
            sum = fmaf(Asub[ty][k], Bsub[k][tx], sum);
        }

        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = sum;
}














__global__ void matrixMulKernel_shared(float* A, float* B, float* C, int n) {
    // Allocate shared memory
    __shared__ float Asub[TILE][TILE];
    __shared__ float Bsub[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = ty + blockIdx.y * blockDim.y;
    int col = tx + blockIdx.x * blockDim.x;

    float sum = 0.0f;

    for (int i = 0; i < (n + blockDim.x - 1) / blockDim.x; i++) {
        if (row < n && (i * blockDim.x + tx) < n)
            Asub[ty][tx] = A[row * n + i * blockDim.x + tx];
        else
            Asub[ty][tx] = 0.0f;

        if (col < n && (i * blockDim.y + ty) < n)
            Bsub[ty][tx] = B[(i * blockDim.y + ty) * n + col];
        else
            Bsub[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < blockDim.x; k++) {
            sum += Asub[ty][k] * Bsub[k][tx];
            // Use fused multiply-add
            //sum = fmaf(Asub[ty][k], Bsub[k][tx], sum);
        }

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

__global__ void matrixMulKernel_naive(float *A, float *B, float *C, int width) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < width && col < width) {
        float value = 0;
        for (int k = 0; k < width; k++) {
            value += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = value;
    }
}

__global__ void matrixMulKernel_row(float *A, float *B, float *C, int width) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < width) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int i = 0; i < width; i++) {
                sum += A[row * width + i] * B[i * N + col];
            }
            C[row * width + col] = sum;
        }
    }
}

__global__ void matrixMulKernel_col(float *A, float *B, float *C, int width) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col < width) {
        for (int row = 0; row < width; row++) {
            float sum = 0.0f;
            for (int i = 0; i < width; i++) {
                sum += A[row * width + i] * B[i * N + col];
            }
            C[row * width + col] = sum;
        }
    }
}

void matrixMulCpu(float *A, float *B, float *C, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float value = 0;
            for (int k = 0; k < width; k++) {
                value += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = value;
        }
    }
}

void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool checkResults(float *A, float *B, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(A[i] - B[i]) > 1e-4) {
            std::cout << "fabs(A[i])=" << fabs(A[i]) << std::endl;
            std::cout << "B[i]=" << B[i] << std::endl;
            std::cout << std::fixed << std::showpoint;
            std::cout << std::setprecision(15);
            std::cout << "fabs(A[i] - B[i])=" << fabs(A[i] - B[i]) << std::endl;
            std::cout << "#####" << std::endl;
            return false;
        }
    }
    return true;
}


int main() {
    srand(static_cast<unsigned int>(time(0)));

    int matrixSize = N * N * sizeof(float);

    float *h_A = (float*)malloc(matrixSize);
    float *h_B = (float*)malloc(matrixSize);
    float *h_C_CPU = (float*)malloc(matrixSize);
    float *h_C_GPU = (float*)malloc(matrixSize);

    initializeMatrix(h_A, N * N);
    initializeMatrix(h_B, N * N);

    auto startCpu = std::chrono::high_resolution_clock::now();
    matrixMulCpu(h_A, h_B, h_C_CPU, N);
    auto stopCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = (stopCpu - startCpu);

    std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl;

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTimeToCopy = 0;
    cudaEventElapsedTime(&gpuTimeToCopy, start, stop);
    std::cout << "GPU memory copy time: " << gpuTimeToCopy << " ms" << std::endl;

    cudaEventRecord(start);

    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // for those kernels we need a different launch configuration
    // int threads = 256;
    // int blocks = (N + threads - 1) / threads ;
    // matrixMulKernel_row<<<blocks, threads>>>(d_A, d_B, d_C, N);
    // matrixMulKernel_col<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "GPU execution time: " << gpuTime << " ms" << std::endl;

    cudaEventRecord(start);

    cudaMemcpy(h_C_GPU, d_C, matrixSize, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);


    cudaEventSynchronize(stop);
    float gpuTimeToRetrieve = 0;
    cudaEventElapsedTime(&gpuTimeToRetrieve, start, stop);
    std::cout << "GPU memory retrieve time: " << gpuTimeToRetrieve << " ms" << std::endl;
    std::cout << "GPU total time: " << (gpuTime + gpuTimeToCopy + gpuTimeToRetrieve)  << " ms" << std::endl;

    double cpuTotalTime = cpuDuration.count();
    double gpuTotalTime = (gpuTime + gpuTimeToCopy + gpuTimeToRetrieve);
    std::cout << "speed up: " << cpuTotalTime / gpuTotalTime << std::endl;

    if (checkResults(h_C_CPU, h_C_GPU, N * N)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
