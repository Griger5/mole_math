#include "cuda_kernel_matrix_operations.cuh"

#include "../../../include/mole_math/macros.h"

const int threads_per_block = 512;

__global__ void cuda_kernel_matrix_determinant_check(double *matrix_values, int *is_zero, size_t i, size_t N) {
    if (matrix_values[i + i*N] == 0) {
        *is_zero = 1;
    }
}

__global__ void cuda_kernel_matrix_determinant_zero_column(double *matrix_values, size_t N, size_t col_num) {
    size_t block_id = blockIdx.x;

    size_t index = threadIdx.x;
    size_t stride = blockDim.x;

    if (block_id < N) {
        size_t row = block_id + col_num + 1;

        double ratio = matrix_values[col_num + row * N] / matrix_values[col_num + col_num * N];

        for (size_t i = index + col_num; i < N; i += stride) {
            matrix_values[i + row * N] -= ratio * matrix_values[i + col_num * N];
        }
    }
}