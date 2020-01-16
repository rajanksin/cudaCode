#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

__global__ void kernelInfo(float* d_A, int width, int height){
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  int offset = iy*width + ix;

  if (ix < width && iy < height){
    printf(" x: %d y: %d offset: %d value: %f\n", ix, iy, offset , d_A[offset]);
  }
}

int main(){

  float *h_A, *d_A;
  int WIDTH = 3;
  int HEIGHT = 2;

  int nElem = WIDTH*HEIGHT;

  h_A = (float*) malloc(sizeof(float)*nElem);

  for(int i =0; i < nElem; i++){
    h_A[i] = i;
  }
  
  // priting on CPU 
  for(int j =0; j < HEIGHT; j++){
    for(int i =0; i< WIDTH; i++){
    printf("%f ,(%d,%d)", h_A[j*WIDTH+i], i,j);}
    printf("\n");
  }


  cudaMalloc((void**)&d_A, sizeof(float)*nElem);

  cudaMemcpy(d_A, h_A, sizeof(float)*nElem, cudaMemcpyHostToDevice);

  dim3 block(WIDTH,HEIGHT);
  dim3 grid((WIDTH+block.x-1)/block.x, (HEIGHT+block.y-1)/block.y);

  kernelInfo<<<grid,block>>>(d_A, WIDTH, HEIGHT);
  cudaDeviceSynchronize();

  free(h_A);
  cudaFree(d_A);
  
}