#include "cuda_kernel_matrix_transform.cuh"

#include "../../../include/mole_math/macros.h"

const int threads_per_block = 512;
const int block_size = 32;

__global__ void cuda_kernel_matrix_subtract_rows(double *row_minuend_values, double *row_subtrahend_values, double multiplier, size_t cols) {
    size_t index = GLOBAL_IDX_X();
    size_t stride = GLOBAL_STRIDE_X();

    for (size_t i = index; i < cols; i += stride) {
        row_minuend_values[i] -= multiplier * row_subtrahend_values[i];
    }
}

__global__ void cuda_kernel_matrix_transpose(double *matrix_values, double *transposed_values, size_t rows, size_t cols) {
    size_t index_x = GLOBAL_IDX_X();
    size_t index_y = GLOBAL_IDX_Y();

    if (index_x < cols && index_y < rows) {
        transposed_values[index_x * cols + index_y] = matrix_values[index_y * cols + index_x];
    }
}

__global__ void cuda_kernel_matrix_transpose_flip(double *matrix_values, double *transposed_values, size_t rows, size_t cols) {
    size_t index_x = GLOBAL_IDX_X();
    size_t index_y = GLOBAL_IDX_Y();

    if (index_x < cols && index_y < rows) {
        transposed_values[index_x * rows + index_y] = matrix_values[index_y * cols + index_x];
    }
}