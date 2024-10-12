#include "cuda_kernel_matrix_operations.cuh"

#include "../../../include/mole_math/macros.h"

__global__ void cuda_kernel_matrix_multiply(double *matrix_a_values, double *matrix_b_values, double *result, size_t rows_a, size_t cols_a, size_t cols_b) {
    __shared__ double shared_a[block_size][block_size];
    __shared__ double shared_b[block_size][block_size];

    size_t index_x = GLOBAL_IDX_X();
    size_t index_y = GLOBAL_IDX_Y();

    size_t local_id_x = threadIdx.x;
    size_t local_id_y = threadIdx.y;

    double temp = 0;
    size_t a_index;
    size_t b_index;
    
    for (int i = 0; i <= cols_a/block_size; i++) {
        a_index = i * block_size + local_id_x;
        b_index = i * block_size + local_id_y;

        if (index_y < rows_a && a_index < cols_a)
            shared_a[local_id_y][local_id_x] = matrix_a_values[index_y * cols_a + a_index];
        else
            shared_a[local_id_y][local_id_x] = 0;
        
        if (index_x < cols_b && b_index < cols_a)
            shared_b[local_id_y][local_id_x] = matrix_b_values[b_index * cols_b + index_x];
        else
            shared_b[local_id_y][local_id_x] = 0;

        __syncthreads();

        for (int k = 0; k < block_size; k++) { 
            temp += shared_a[local_id_y][k] * shared_b[k][local_id_x]; 
        }

        __syncthreads();
    }

    if (index_y < rows_a && index_x < cols_b)
        result[index_y*cols_b+index_x] = temp;
}

__global__ void cuda_kernel_matrix_subtract_elements(double *matrix_a_values, double *matrix_b_values, double *result, size_t rows, size_t cols) {
    size_t index = GLOBAL_IDX_X();
    size_t stride = GLOBAL_STRIDE_X();
    
    for (size_t i = index; i < rows*cols; i += stride) {
        result[i] = matrix_a_values[i] - matrix_b_values[i]; 
    }
}

__global__ void cuda_kernel_matrix_multiply_elements(double *matrix_a_values, double *matrix_b_values, double *result, size_t rows, size_t cols) {
    size_t index = GLOBAL_IDX_X();
    size_t stride = GLOBAL_STRIDE_X();
    
    for (size_t i = index; i < rows*cols; i += stride) {
        result[i] = matrix_a_values[i] * matrix_b_values[i]; 
    }
}