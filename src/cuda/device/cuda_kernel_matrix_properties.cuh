#ifndef CUDA_KERNEL_MATRIX_PROPERTIES_H
#define CUDA_KERNEL_MATRIX_PROPERTIES_H

__global__ void cuda_kernel_matrix_determinant_check(double *matrix_values, int *is_zero, size_t i, size_t N);
__global__ void cuda_kernel_matrix_determinant_zero_column(double *matrix_values, size_t N, size_t col_num);

#endif