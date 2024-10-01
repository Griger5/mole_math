#include <stdio.h>

#include "cuda_kernel_matrix_transform.cuh"

#include "../../../include/mole_math/macros.h"

__global__ void cuda_kernel_matrix_subtract_rows(double *row_minuend_values, double *row_subtrahend_values, double multiplier, size_t cols) {
    size_t index = GLOBAL_IDX_X();
    size_t stride = GLOBAL_STRIDE_X();

    for (size_t i = index; i < cols; i += stride) {
        row_minuend_values[i] -= multiplier * row_subtrahend_values[i];
    }
}