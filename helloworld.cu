#include <stdio.h>

__global__ void hellofromGPU()
{
    printf("Hello from GPU:%d\n", threadIdx.x);
}

int main()
{
    printf("Hello from CPU\n");

    hellofromGPU<<<1,10>>>();

    cudaDeviceSynchronize();
    return 0;
}