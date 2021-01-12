

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

void reduction_gold(float* odata, float* idata, const unsigned int len) 
{
  *odata = 0;
  for(int i=0; i<len; i++) *odata += idata[i];
}

////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata, int elements_per_block, int min_power2)
{
    // dynamically allocated shared memory

    extern  __shared__  float temp[];

    int tid = threadIdx.x;

    // first, each thread loads data into shared memory

    temp[tid] = g_idata[elements_per_block * blockIdx.x + tid];

    // next, we perform binary tree reduction

    for (int d = min_power2 >> 1; d > 0; d >>= 1) {
      __syncthreads();  // ensure previous step completed 
      if (tid < d && tid + d < blockDim.x)  temp[tid] += temp[tid+d];
    }

    // finally, first thread puts result into global memory

    if (tid==0) g_odata[blockIdx.x] = temp[0];
}


__global__ void warp_reduction(float *g_odata, float *g_idata, int elements_per_block, int min_power2)
{
    // dynamically allocated shared memory

    extern  __shared__  float temp[];

    int tid = threadIdx.x;

    // first, each thread loads data into shared memory

    temp[tid] = g_idata[elements_per_block * blockIdx.x + tid];

    // next, we perform binary tree reduction

    int thread_offset = (tid / 32) * 32;

    for (int i = 16; i > 0; i /= 2) {
        temp[tid] += __shfl_down_sync(-1, temp[tid], thread_offset + i);
    }

    if (tid % 32 == 0) {
        for (int d = min_power2 >> 1; d >= 32; d >>= 1) {
            __syncthreads();
            if (tid < d && tid + d < blockDim.x)  temp[tid] += temp[tid+d];
        }
    }

    // finally, first thread puts result into global memory

    if (tid==0) g_odata[blockIdx.x] = temp[0];
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_blocks, num_elements, num_threads, mem_size, shared_mem_size, min_power2;

  float *h_data, *reference, sum;
  float *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  num_threads  = 125;
  num_elements = num_threads * 32;
  num_blocks = num_elements / num_threads;
  mem_size     = sizeof(float) * num_elements;

  min_power2 = 1;
  while (min_power2 < num_threads) min_power2 <<= 1;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 1000

  h_data = (float*) malloc(mem_size);
      
  for(int i = 0; i < num_elements; i++) 
    h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));

  // compute reference solutions

  reference = (float*) malloc(mem_size);
  reduction_gold(&sum, h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, num_blocks * sizeof(float)) );

  // copy host memory to device input array

  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );

  // execute the kernel

  shared_mem_size = sizeof(float) * num_elements / num_blocks;
  warp_reduction<<<num_blocks,num_threads,shared_mem_size>>>(d_odata, d_idata, num_elements / num_blocks, min_power2);
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host

  checkCudaErrors( cudaMemcpy(h_data, d_odata, sizeof(float) * num_blocks, cudaMemcpyDeviceToHost) );

  // check results
  float d_sum = 0;
  for (int i = 0; i < num_blocks; i++) {
      d_sum += h_data[i];
  }
  printf("reduction error = %f\n", d_sum - sum);

  // cleanup memory

  free(h_data);
  free(reference);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
