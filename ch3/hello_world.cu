#include <iostream> 

 __global__ void helloWorld() { 

   int tid = threadIdx.x + blockIdx.x * blockDim.x; 

   printf("Hello, World! Thread %d\n", tid); 

 } 

 int main() { 

   helloWorld<<<1, 10>>>();

   cudaDeviceSynchronize();

   return 0; 

 } 