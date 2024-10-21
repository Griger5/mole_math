#ifndef CUDA_KERNEL_MATRIX_UTILS_H
#define CUDA_KERNEL_MATRIX_UTILS_H

__global__ void cuda_kernel_matrix_identity(double *result, size_t N);

__global__ void cuda_kernel_matrix_random(double *result, size_t size);

__global__ void cuda_kernel_matrix_init_integers(double *result, size_t size);

#endif