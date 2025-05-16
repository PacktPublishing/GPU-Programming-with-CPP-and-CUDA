## Hello World
Here is the command to compile from the terminal:
nvcc hello_world.cu -o hello_world -Xcompiler "-Wall" -lcudadevrt -lcudart_static -lstdc++

or if you are using the docker terminal:
docker run -it --rm --gpus all -v ./:/code nvidia/cuda:12.0.0-devel-ubuntu20.04 bash 
cd code
nvcc -o hello_world hello_world.cu

## Device Query
Here is the command to compile from the terminal:
nvcc -o device_query device_query.cu -Xcompiler "-Wall" -lcudadevrt -lcudart_static -lstdc++

or if you are using the docker terminal:
docker run -it --rm --gpus all -v ./:/code nvidia/cuda:12.0.0-devel-ubuntu20.04 bash 
cd code
nvcc -o device_query device_query.cu
