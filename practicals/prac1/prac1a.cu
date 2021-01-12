// The -ptx option caused that PTX instructions were generated instead of the binary. These instructions are used by
// a graphics driver that translates PTX instructions into the binary code.
//
// One interesting phenomenon is that even if we access invalid elements of an array, the program doesn't show any
// errors. Even if the subsequent instructions will not be executed because of an error, our program neither crashes nor
// reports an error. It means that we need to use special methods like 'getLastCudaError' to catch errors that
// encountered on the kernel side.

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>

//
// kernel routine
// 

__global__ void my_first_kernel(float *x, float *y, float *z)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  z[tid] = x[tid] + y[tid];
}


//
// main code
//

int main(int argc, char **argv)
{
  float *h_x, *d_x, *h_y, *d_y, *h_z, *d_z;
  int   nblocks, nthreads, nsize, n; 

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads;

  // allocate memory for array

  h_x = (float *)malloc(nsize*sizeof(float));
  h_y = (float *)malloc(nsize*sizeof(float));
  h_z = (float *)malloc(nsize*sizeof(float));

  for (n=0; n<nsize; n++) h_x[n] = n;
  for (n=0; n<nsize; n++) h_y[n] = 1.0 / (n + 1);

  cudaMalloc((void **)&d_x, nsize*sizeof(float));
  cudaMalloc((void **)&d_y, nsize*sizeof(float));
  cudaMalloc((void **)&d_z, nsize*sizeof(float));

  cudaMemcpy(d_x, h_x, nsize*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, nsize*sizeof(float), cudaMemcpyHostToDevice);

  // execute kernel

  my_first_kernel<<<nblocks,nthreads>>>(d_x, d_y, d_z);

  // copy back results and print them out

  cudaMemcpy(h_z, d_z, nsize*sizeof(float), cudaMemcpyDeviceToHost);

  for (n=0; n<nsize; n++) printf("h_x[%d] = %f \n", n, h_x[n]);
  for (n=0; n<nsize; n++) printf("h_y[%d] = %f \n", n, h_y[n]);
  for (n=0; n<nsize; n++) printf("h_z[%d] = %f \n", n, h_z[n]);

  // free memory 

  cudaFree(d_x);
  free(h_x);
  cudaFree(d_y);
  free(h_y);
  cudaFree(d_z);
  free(h_z);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
