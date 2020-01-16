#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>


void allocateCPUMemory(float **h_A, float** h_B, float** h_C, int size){

  printf("CPU memory allocating ...\n");
  *h_A =(float*) malloc(sizeof(float)*size*size);
  *h_B =(float*) malloc(sizeof(float)*size*size);
  *h_C =(float*) malloc(sizeof(float)*size*size);

  printf("CPU memory allocated...\n");

}

void allocateGPUMemory(float **d_A, float** d_B, float** d_C, int size){

   cudaMalloc(((void**)d_A),  sizeof(float)*size*size);
   cudaMalloc(((void**)d_B),  sizeof(float)*size*size);
   cudaMalloc(((void**)d_C),  sizeof(float)*size*size);

}

void initCPUData(float* h_A, float* h_B, int size){

  for(int i =0; i < size; i++){
    for(int j=0; j < size; j++){
      int offset = i*size+j;
      // printf("offset: %d\n", offset);
      *(h_A+offset) = offset;
      *(h_B+offset) = 2*offset;
    }
  }
}

void printdata(float* x, int size){

  for(int i =0; i < size; i++){
    for(int j=0; j< size; j++){
      int offset = i*size+j;
      printf("%f ", x[offset]);
    }
    printf("\n");
  }

}


__global__ void AddKernel(float* d_A, float* d_B, float* d_C, int size){

  // printf("GPU ...........");
  int row = threadIdx.y + blockIdx.y*blockDim.y ;
  int col = threadIdx.x + blockIdx.x*blockDim.x ;
  // printf("blockDim.x: %d, blockIdx.x: %d , threadIdx.x:%d \n", blockDim.x, blockIdx.x, threadIdx.x);
  // printf("blockDim.y: %d, blockIdx.y: %d , threadIdx.y:%d \n", blockDim.y, blockIdx.y, threadIdx.y);
  int offset = row*size + col;
  
  if(row < size && col < size){
    printf("threadIdx.x:%d threadIdx.y:%d ---> row: %d col:%d\n", threadIdx.x, threadIdx.y, row, col);
    // printf("Kernel played for : row: %d col: %d \n", row, col);
    // printf("row: %d , col %d, idx: %d\n", row, col, offset);
    d_C[offset]= d_A[offset] + d_B[offset];
  }
}

__global__ void AddPerRowKernel(float* d_A, float* d_B, float* d_C, int size){

  int row = threadIdx.y + blockIdx.y*blockDim.y ;
  // int col = threadIdx.x + blockIdx.x*blockDim.x ;

  if(row < size){
    printf("Kernel played for : row: %d\n", row);
    for(int col =0; col < size; col++){
      int offset = row*size + col;
      d_C[offset] = d_A[offset] + d_B[offset];
    }
  }
}

__global__  void printData(float* d_A){
  int size = 4;
  int row = threadIdx.y + blockIdx.y*blockDim.y;
  int col = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = row*size + col;
  if(row < size && col < size){
    printf("offset: %d , value: %f", offset, d_A[offset]);

  } 
} 

double cpuSecond() {
 struct timeval tp;
 gettimeofday(&tp,NULL);
 return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


int main(){

  float* h_A, *d_A;
  float* h_B, *d_B;
  float* h_C, *d_C;
  int size = 4; 

  int nElements = size*size;

  allocateCPUMemory(&h_A, &h_B, &h_C, size);
  allocateGPUMemory(&d_A, &d_B, &d_C, size);

  // cudaMalloc(((void**)&d_A),  sizeof(float)*size*size);
  //  cudaMalloc(((void**)&d_B),  sizeof(float)*size*size);
  //  cudaMalloc(((void**)&d_C),  sizeof(float)*size*size);

  printf("allocation done\n");
  initCPUData(h_A, h_B, size);

  // printf(" ------------------ Data A --------------------------\n");
  // printdata(h_A, size);
  // printf(" ------------------ Data B --------------------------\n");
  // printdata(h_B, size);

  cudaMemcpy(d_A, h_A, nElements*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nElements*sizeof(float), cudaMemcpyHostToDevice);

  // // kernel call
  dim3 block(32,32,1);
  dim3 grid(size+block.x-1/block.x,size+block.y-1/block.y,1);
   
  AddKernel<<<grid,block>>>(d_A, d_B, d_C, size);
  // AddPerRowKernel<<<grid,block>>>(d_A, d_B, d_C, size);
  // printData<<<1,block>>>(d_A);
  cudaDeviceSynchronize();

  // // copying result back;
  cudaMemcpy(h_C, d_C, nElements*sizeof(float), cudaMemcpyDeviceToHost);

  // printf(" ------------------ Data C --------------------------\n");
  // printdata(h_C, size);

  //verify data
  bool veri_pass = true;
  for(int i =0; i < size ; i++){
    for(int j =0; j < size; j++){
      int idx = i*size + j;
      if(h_C[idx] != (h_A[idx] + h_B[idx])){
        veri_pass = false;
        break;
      }
    }
    if(!veri_pass) break;
  }

  if(veri_pass){
    printf("Verification pass .......");
  }
  else {
    printf("verification failed .......");
  }

  cudaFree(d_A);
  cudaFree(d_B);

  free(h_A);
  free(h_B);
  free(h_C);


  cudaDeviceReset();

  return 0;
}