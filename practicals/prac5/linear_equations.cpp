#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>


void print_square_matrix(float *matrix, int rows_count) {
    for (int i = 0; i < rows_count * rows_count; i += rows_count) {
        for (int j = 0; j < rows_count; j++) {
            printf("%.3f ", matrix[i + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_vector(float *vector_array, int elements_count) {
    for (int i = 0; i < elements_count; i++) {
        printf("%.2f\n", vector_array[i]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    const int VECTOR_ELEMENTS_COUNT = 2;
    const int MATRIX_ELEMENTS_COUNT = VECTOR_ELEMENTS_COUNT * VECTOR_ELEMENTS_COUNT;

    float **d_matrix_pointer, **d_inversed_matrix_pointer;
    float *h_matrix, *d_matrix, *h_inversed_matrix, *d_inversed_matrix;
    float *h_x_vector, *d_x_vector, *h_y_vector, *d_y_vector;
    int *dlu_pivots, *dlu_info;
    float alpha_parameter = 1.0f, beta_parameter = 0.0f;

    h_matrix = (float*)malloc(MATRIX_ELEMENTS_COUNT * sizeof(float));
    h_inversed_matrix = (float*)malloc(MATRIX_ELEMENTS_COUNT * sizeof(float));
    h_x_vector = (float*)malloc(VECTOR_ELEMENTS_COUNT * sizeof(float));
    h_y_vector = (float*)malloc(VECTOR_ELEMENTS_COUNT * sizeof(float));
    for (int i = 0; i < MATRIX_ELEMENTS_COUNT; i++) {
        h_matrix[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < VECTOR_ELEMENTS_COUNT; i++) {
        h_y_vector[i] = rand() / (float)RAND_MAX;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMalloc(&d_matrix_pointer, sizeof(float*));
    cudaMalloc(&d_inversed_matrix_pointer, sizeof(float*));
    cudaMalloc(&d_matrix, sizeof(float) * MATRIX_ELEMENTS_COUNT);
    cudaMalloc(&d_inversed_matrix, sizeof(float) * MATRIX_ELEMENTS_COUNT);
    cudaMalloc(&d_x_vector, sizeof(float) * VECTOR_ELEMENTS_COUNT);
    cudaMalloc(&d_y_vector, sizeof(float) * VECTOR_ELEMENTS_COUNT);
    cudaMalloc(&dlu_pivots, VECTOR_ELEMENTS_COUNT * sizeof(int));
    cudaMalloc(&dlu_info, sizeof(int));

    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * MATRIX_ELEMENTS_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_vector, h_y_vector, sizeof(float) * VECTOR_ELEMENTS_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_pointer, &d_matrix, sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inversed_matrix_pointer, &d_inversed_matrix, sizeof(float*), cudaMemcpyHostToDevice);

    cublasSgetrfBatched(
        handle, VECTOR_ELEMENTS_COUNT, d_matrix_pointer, VECTOR_ELEMENTS_COUNT, dlu_pivots, dlu_info, 1
    );
    cudaDeviceSynchronize();
    cublasSgetriBatched(
        handle, VECTOR_ELEMENTS_COUNT, (const float**) d_matrix_pointer, VECTOR_ELEMENTS_COUNT, dlu_pivots,
        d_inversed_matrix_pointer, VECTOR_ELEMENTS_COUNT, dlu_info, 1
    );
    cudaDeviceSynchronize();
    cublasSgemv(
        handle, CUBLAS_OP_T, VECTOR_ELEMENTS_COUNT, VECTOR_ELEMENTS_COUNT, &alpha_parameter, d_inversed_matrix,
        VECTOR_ELEMENTS_COUNT, d_y_vector, 1, &beta_parameter, d_x_vector, 1
    );
    cudaDeviceSynchronize();
    cudaMemcpy(h_inversed_matrix, d_inversed_matrix, sizeof(float) * MATRIX_ELEMENTS_COUNT, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x_vector, d_x_vector, sizeof(float) * VECTOR_ELEMENTS_COUNT, cudaMemcpyDeviceToHost);

    printf("Matrix of equation:\n");
    print_square_matrix(h_matrix, VECTOR_ELEMENTS_COUNT);
    printf("Inversed matrix of equation:\n");
    print_square_matrix(h_inversed_matrix, VECTOR_ELEMENTS_COUNT);
    printf("Elements of X elements:\n");
    print_vector(h_x_vector, VECTOR_ELEMENTS_COUNT);
    printf("Elements of Y elements:\n");
    print_vector(h_y_vector, VECTOR_ELEMENTS_COUNT);
    return 0;
}


