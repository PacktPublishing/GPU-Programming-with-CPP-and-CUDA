# GPU-Programming-with-C-and-CUDA
GPU Programming with C++ and CUDA, published by Packt

# Code conventions

We are using the following convetions:

1. camelCase for names of functions, kernels and variables
2. CamelCase with uppercase letter at beginning for structs
3. snake_case for names of files
4. When using two functions to perform the same computation on the CPU and GPU we use the same name with a Cpu/Kernel suffix like: computeSomethingCpu / computeSomethingKernel
5. When we need to allocate buffers with similar names, we are using h_ preffix for host side and d_for device side.
    * float* h_A;
    * float* d_A;
6. When comparing results we add a suffix to the name of the variable _CPU or _GPU.
    * float* h_C_GPU; // this is the C array or matrix calculated on the GPU and copied back to the host
    * float* h_C_CPU; // this is the C array or matrix calculated on the CPU
