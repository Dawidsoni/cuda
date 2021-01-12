// If the synchronization is not be performed, values of the x array are not be updated by the kernel.

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


__device__ __managed__ double *x;
//
// kernel routine
// 

__global__ void my_first_kernel()
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = (float) threadIdx.x;
}


//
// main code
//

int main(int argc, const char **argv)
{
  int   nblocks, nthreads, nsize, n;

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  checkCudaErrors(cudaMallocManaged(&x, nsize*sizeof(float)));

  // execute kernel

  my_first_kernel<<<nblocks,nthreads>>>();
  getLastCudaError("my_first_kernel execution failed\n");

  cudaDeviceSynchronize();

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,x[n]);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
