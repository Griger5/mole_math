#include "cuda_kernel_matrix_funcs.cuh"

#include "../../../include/mole_math/macros.h"

const int threads_per_block = 512;

__global__ void cuda_kernel_matrix_sum_row(double *row_values, size_t cols, double *result) {
    extern __shared__ double shared_values[];
    int local_id = threadIdx.x;
    size_t global_id = GLOBAL_IDX_X();
    size_t stride = GLOBAL_STRIDE_X();

    shared_values[local_id] = 0.0;
    
    for (size_t i = global_id; i < cols; i += stride) {
        shared_values[local_id] += row_values[global_id];
    }

    for (size_t s = blockDim.x/2; s>0; s>>=1) {
        __syncthreads();

        if (local_id < s)
            shared_values[local_id] += shared_values[local_id + s];
    }

    // atomicAdd(double *, double) isn't available on all architectures
    if (local_id == 0) result[blockIdx.x] = shared_values[0];
}