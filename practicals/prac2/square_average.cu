#include <iostream>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

// Task 5:
// In the case of the slower approach, the effective transfer rate was estimated to 5 Gb/s, while in the case of the
// faster approach, it was estimated to about 20 Gb/s. As Tesla graphic card was used in the conducted experiments, the
// theoretical peak capability was equal to 80 Gb/s.

__constant__ int a_value, b_value, c_value, thread_elements;

__global__ void calculate_average(float* dev_inputs, float* dev_outputs) {
    int output_index = threadIdx.x + blockIdx.x * blockDim.x;
    int first_input_index = output_index * thread_elements;
    float values_sum = 0;
    for (int i = first_input_index; i < first_input_index + thread_elements; i++) {
        values_sum += a_value * dev_inputs[i] * dev_inputs[i] + b_value * dev_inputs[i] + c_value;
    }
    dev_outputs[output_index] = values_sum / thread_elements;
}

int main(int argc, const char** argv) {
    const int host_a_value = 3.0;
    const int host_b_value = 5.0;
    const int host_c_value = 1.0;
    const int host_inputs_count = 640000;
    const int host_outputs_count = 6400;
    const int host_thread_elements = host_inputs_count / host_outputs_count;
    checkCudaErrors(cudaMemcpyToSymbol(a_value, &host_a_value, sizeof(host_a_value)));
    checkCudaErrors(cudaMemcpyToSymbol(b_value, &host_b_value, sizeof(host_b_value)));
    checkCudaErrors(cudaMemcpyToSymbol(c_value, &host_c_value, sizeof(host_c_value)));
    checkCudaErrors(cudaMemcpyToSymbol(thread_elements, &host_thread_elements, sizeof(host_thread_elements)));

    float* dev_inputs, *dev_outputs, *host_outputs;
    checkCudaErrors(cudaMalloc((void**)&dev_inputs, sizeof(float) * host_inputs_count));
    checkCudaErrors(cudaMalloc((void**)&dev_outputs, sizeof(float) * host_outputs_count));
    host_outputs = (float*)malloc(sizeof(float) * host_outputs_count);

    curandGenerator_t random_generator;
    checkCudaErrors(curandCreateGenerator(&random_generator, CURAND_RNG_PSEUDO_DEFAULT));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(random_generator, 1234ULL));
    checkCudaErrors(curandGenerateNormal(random_generator, dev_inputs, host_inputs_count, 0.0f, 1.0f));

    calculate_average<<<200, 32>>>(dev_inputs, dev_outputs);
    checkCudaErrors(cudaMemcpy(host_outputs, dev_outputs, sizeof(float) * host_outputs_count, cudaMemcpyDeviceToHost));
    float outputs_sum = 0;
    for (int i = 0; i < host_outputs_count; i++) {
        outputs_sum += host_outputs[i];
    }
    std::cout << "\n\n/////////////////////////////////////////////////////////////\n";
    std::cout << "Mean: " << (outputs_sum / static_cast<float>(host_outputs_count)) << "\n\n";

    checkCudaErrors(cudaFree(dev_inputs));
    checkCudaErrors(cudaFree(dev_outputs));
    free(host_outputs);
}