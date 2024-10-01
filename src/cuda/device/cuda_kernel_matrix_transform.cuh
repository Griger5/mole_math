#ifndef CUDA_KERNEL_MATRIX_TRANSFORM_H
#define CUDA_KERNEL_MATRIX_TRANSFORM_H

const int threads_per_block = 512;

__global__ void cuda_kernel_matrix_subtract_rows(double *row_minuend_values, double *row_subtrahend_values, double multiplier, size_t cols);

#endif