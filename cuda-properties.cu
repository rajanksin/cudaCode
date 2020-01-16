#include <stdio.h>
#include <cuda_runtime.h>


__global__ void hellofromGPU(void){
  printf("Hello from GPU\n");
}

void getDevProperties(){
  int dev_count;
  cudaGetDeviceCount(&dev_count);
  printf("Device count: %d\n", dev_count);
  cudaDeviceProp dev_prop;
  for(int i =0; i < dev_count; i++){
    cudaGetDeviceProperties(&dev_prop, i);
    printf("Device #%d\n", i);
    printf("Total global memory in GB: %f\n", dev_prop.totalGlobalMem/(1024.0*1024*1024));
    printf("maxThreadsPerBlock: %d\n", dev_prop.maxThreadsPerBlock);
    printf("multiProcessorCount: %d\n", dev_prop.multiProcessorCount);
    printf("maxBlockDim x: %d, y :%d, z:%d\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
    printf("maxThreadDim x: %d, y :%d, z:%d\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    printf("warpSize :%d\n", dev_prop.warpSize);
    printf("maxThreadPerBlock: %d\n", dev_prop.maxThreadsPerBlock);
    printf("sharedMemPerBlock: %zd\n", dev_prop.sharedMemPerBlock);
    printf("regsPerBlock: %d\n", dev_prop.regsPerBlock);
    printf("sm: major : %d, minor: %d\n", dev_prop.major, dev_prop.minor);



  }
}

int main(){
  printf("hello from CPU ....\n");
  // hellofromGPU<<<1,10>>>();

  getDevProperties();
  cudaDeviceReset();
  return 0;
}