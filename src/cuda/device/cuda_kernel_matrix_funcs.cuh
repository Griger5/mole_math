#ifndef CUDA_KERNEL_MATRIX_TRANSFORM_H
#define CUDA_KERNEL_MATRIX_TRANSFORM_H

__global__ void cuda_kernel_matrix_sum_row(double *row_values, size_t cols, double *result);

#endif