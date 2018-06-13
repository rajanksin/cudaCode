// #include <stdlib>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void sumArrayonHost(float* A, float* B, float* C, int N)
{

    int i = threadIdx.x;
    if(i < N)
        C[i] = A[i] + B[i];

}

void initData(float* data, int size)
{
    for(int i=0; i < size; i++)
    {
        data[i] = (float) (rand() & 0xFF)/10.0f;
    }
}

#define CHECK(call) {const cudaError_t error = call;  if (error != cudaSuccess) { printf("Error: %s:%d, ", __FILE__, __LINE__);  printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  exit(1);}}

int main()
{
    float *d_A, *d_B, *d_C;
    int n_ele =1 << 24;
    dim3 block(512);
    dim3  grid((block.x-1+n_ele)/block.x);
    int bytes = n_ele * sizeof(float);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    cudaMalloc((void**)&d_A, n_ele*sizeof(float));
    cudaMalloc((void**)&d_B, n_ele*sizeof(float));
    cudaMalloc((void**)&d_C, n_ele*sizeof(float));

    float *h_A, *h_B, *h_C;
    h_A = (float*) malloc (bytes);
    h_B = (float*) malloc (bytes);
    h_C = (float*) malloc (bytes);

    initData(h_A, n_ele);
    initData(h_B, n_ele);
    initData(h_C, n_ele);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);

    printf("grid:%d  block:%d\n", grid.x, block.x);
    sumArrayonHost<<<grid,block>>>(d_A, d_B, d_C, n_ele);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // for(int i =0; i< n_ele; i++)
    //     printf("%f ", h_A[i]);
    // printf("\n");
    // for(int i =0; i< n_ele; i++)
    //     printf("%f ", h_B[i]);
    // printf("\n");
    // for(int i =0; i< n_ele; i++)
    //     printf("%f ", h_C[i]);

    printf("\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}