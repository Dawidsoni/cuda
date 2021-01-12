#include <iostream>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>


__constant__ int a_value, b_value, c_value, thread_elements;

// The experiments show that computing floats is 2-4 times faster than computing doubles on GPUs.

template <class T>
__global__ void calculate_average(T* inputs, T* outputs) {
    int output_index = threadIdx.x + blockIdx.x * blockDim.x;
    int first_input_index = output_index * thread_elements;
    float values_sum = 0;
    for (int i = first_input_index; i < first_input_index + thread_elements; i++) {
        float a_power = a_value, b_power = b_value, c_power = c_value;
        for (int power = 1; power <= 2; power++) {
            values_sum += a_power * inputs[i] * inputs[i] + b_power * inputs[i] + c_power;
            a_power *= a_value;
            b_power *= b_value;
            c_power *= c_value;
        }
    }
    outputs[output_index] = values_sum / thread_elements;
}


int main(int argc, const char** argv) {
    const int host_a_value = 2;
    const int host_b_value = 3;
    const int host_c_value = 4;
    const int host_inputs_count = 640000;
    const int host_outputs_count = 6400;
    const int host_thread_elements = host_inputs_count / host_outputs_count;
    cudaEvent_t start, stop;
    float milli;

    checkCudaErrors(cudaMemcpyToSymbol(a_value, &host_a_value, sizeof(host_a_value)));
    checkCudaErrors(cudaMemcpyToSymbol(b_value, &host_b_value, sizeof(host_b_value)));
    checkCudaErrors(cudaMemcpyToSymbol(c_value, &host_c_value, sizeof(host_c_value)));
    checkCudaErrors(cudaMemcpyToSymbol(thread_elements, &host_thread_elements, sizeof(host_thread_elements)));

    float* dev_float_inputs, *dev_float_outputs, *host_float_outputs;
    checkCudaErrors(cudaMalloc((void**)&dev_float_inputs, sizeof(float) * host_inputs_count));
    checkCudaErrors(cudaMalloc((void**)&dev_float_outputs, sizeof(float) * host_outputs_count));
    host_float_outputs = (float*)malloc(sizeof(float) * host_outputs_count);

    double* dev_double_inputs, *dev_double_outputs, *host_double_outputs;
    checkCudaErrors(cudaMalloc((void**)&dev_double_inputs, sizeof(double) * host_inputs_count));
    checkCudaErrors(cudaMalloc((void**)&dev_double_outputs, sizeof(double) * host_outputs_count));
    host_double_outputs = (double*)malloc(sizeof(double) * host_outputs_count);

    curandGenerator_t random_generator;
    checkCudaErrors(curandCreateGenerator(&random_generator, CURAND_RNG_PSEUDO_DEFAULT));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(random_generator, 1234ULL));
    checkCudaErrors(curandGenerateNormal(random_generator, dev_float_inputs, host_inputs_count, 0.0f, 1.0f));
    checkCudaErrors(curandGenerateNormalDouble(random_generator, dev_double_inputs, host_inputs_count, 0.0L, 1.0L));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    calculate_average<<<200, 32>>>(dev_float_inputs, dev_float_outputs);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    printf("Computation time for floats: %.3f (ms)\n", milli);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    calculate_average<<<200, 32>>>(dev_double_inputs, dev_double_outputs);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    printf("Computation time for doubles: %.3f (ms)\n", milli);
}