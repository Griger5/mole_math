#ifndef CUDA_KERNEL_MATRIX_SCALARS_H
#define CUDA_KERNEL_MATRIX_SCALARS_H

const int threads_per_block = 512;

__global__ void cuda_kernel_matrix_subtract_scalar(double *matrix_values, double scalar, size_t size);

__global__ void cuda_kernel_matrix_multiply_row_scalar(double *row_values, double scalar, size_t size);

#endif