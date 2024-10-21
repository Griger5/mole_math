#include <curand_kernel.h>

#include "cuda_kernel_matrix_utils.cuh"

#include "../../../include/mole_math/macros.h"

const int threads_per_block = 512;

__global__ void cuda_kernel_matrix_identity(double *result, size_t N) {
    size_t index = GLOBAL_IDX_X();
    size_t stride = GLOBAL_STRIDE_X();

    for (size_t i = index; i < N; i += stride) {
        result[i + i * N] = 1.0;
    }
}

__global__ void cuda_kernel_matrix_random(double *result, size_t size) {
    size_t index = GLOBAL_IDX_X();
    size_t stride = GLOBAL_STRIDE_X();
    
    curandState_t rng;
	curand_init(clock64(), index, 0, &rng);

    for (size_t i = index; i < size; i += stride) {
        result[i] = curand_uniform_double(&rng);
    }
}

__global__ void cuda_kernel_matrix_init_integers(double *result, size_t size) {
    size_t index = GLOBAL_IDX_X();
    size_t stride = GLOBAL_STRIDE_X();

    for (size_t i = index; i < size; i += stride) {
        result[i] = i + 1;
    }
}