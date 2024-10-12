#ifndef CUDA_KERNEL_MATRIX_OPERATIONS_H
#define CUDA_KERNEL_MATRIX_OPERATIONS_H

const int threads_per_block = 512;
const int block_size = 32;

__global__ void cuda_kernel_matrix_multiply(double *matrix_a_values, double *matrix_b_values, double *result, size_t rows_a, size_t cols_a, size_t cols_b);

__global__ void cuda_kernel_matrix_subtract_elements(double *matrix_a_values, double *matrix_b_values, double *result, size_t rows, size_t cols);

__global__ void cuda_kernel_matrix_multiply_elements(double *matrix_a_values, double *matrix_b_values, double *result, size_t rows, size_t cols);

#endif