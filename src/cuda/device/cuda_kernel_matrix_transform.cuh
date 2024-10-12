#ifndef CUDA_KERNEL_MATRIX_TRANSFORM_H
#define CUDA_KERNEL_MATRIX_TRANSFORM_H

const int threads_per_block = 512;
const int block_size = 32;

__global__ void cuda_kernel_matrix_subtract_rows(double *row_minuend_values, double *row_subtrahend_values, double multiplier, size_t cols);

__global__ void cuda_kernel_matrix_transpose(double *matrix_values, double *transposed_values, size_t rows, size_t cols);
__global__ void cuda_kernel_matrix_transpose_flip(double *matrix_values, double *transposed_values, size_t rows, size_t cols);

#endif