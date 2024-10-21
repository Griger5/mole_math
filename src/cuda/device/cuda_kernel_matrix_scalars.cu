#include "cuda_kernel_matrix_operations.cuh"

#include "../../../include/mole_math/macros.h"

const int threads_per_block = 512;

__global__ void cuda_kernel_matrix_subtract_scalar(double *matrix_values, double scalar, size_t size) {
    size_t index = GLOBAL_IDX_X();
    size_t stride = GLOBAL_STRIDE_X();

    for (size_t i = index; i < size; i += stride) {
        matrix_values[i] -= scalar;
    }
}

__global__ void cuda_kernel_matrix_multiply_row_scalar(double *row_values, double scalar, size_t cols) {
    size_t index = GLOBAL_IDX_X();
    size_t stride = GLOBAL_STRIDE_X();

    for (size_t i = index; i < cols; i += stride) {
        row_values[i] *= scalar;
    }
}