#include <cuda_runtime.h>
#include <iostream>

struct Point {
    float x;
    float z;
    float y;
};

__global__ void calculateEuclideanDistanceKernel(Point *lineA, Point *lineB, float *distances, int numPoints) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < numPoints) {
        float dx = lineA[idx].x - lineB[idx].x;
        float dy = lineA[idx].y - lineB[idx].y;
        float dz = lineA[idx].z - lineB[idx].z;
        
        distances[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
        
    }
}

int main() {
    int numPoints = 10'000'000;
    size_t sizePoints = numPoints * sizeof(Point);
    size_t sizeDistances = numPoints * sizeof(float);

    Point *h_lineA = (Point *)malloc(sizePoints);
    Point *h_lineB = (Point *)malloc(sizePoints);
    float *h_distances = (float *)malloc(sizeDistances);

    for (int i = 0; i < numPoints; i++) {
        h_lineA[i].x = i * 1.0f;
        h_lineA[i].y = i * 2.0f;
        h_lineA[i].z = i * 3.0f;
        h_lineB[i].x = i * 0.5f;
        h_lineB[i].y = i * 1.5f;
        h_lineB[i].z = i * 2.5f;
    }

    Point *d_lineA;
    Point *d_lineB;
    float *d_distances;
    cudaMalloc((void **)&d_lineA, sizePoints);
    cudaMalloc((void **)&d_lineB, sizePoints);
    cudaMalloc((void **)&d_distances, sizeDistances);

    cudaMemcpy(d_lineA, h_lineA, sizePoints, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lineB, h_lineB, sizePoints, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    calculateEuclideanDistanceKernel<<<gridSize, blockSize>>>(d_lineA, d_lineB, d_distances, numPoints);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float gpuDuration;
    cudaEventElapsedTime(&gpuDuration, start, stop);

    std::cout<< std::fixed << "Time taken: " << gpuDuration << " ms" << std::endl;

    cudaMemcpy(h_distances, d_distances, sizeDistances, cudaMemcpyDeviceToHost);

    cudaFree(d_lineA);
    cudaFree(d_lineB);
    cudaFree(d_distances);

    free(h_lineA);
    free(h_lineB);
    free(h_distances);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
