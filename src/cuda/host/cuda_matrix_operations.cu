#include "../../../include/mole_math/cuda_matrix_operations.cuh"

#include "../device/cuda_kernel_matrix_operations.cuh"

#include "../../../include/mole_math/cuda_check_error.cuh"

#include "../../../include/mole_math/seq_matrix_utils.h"

Matrix cuda_matrix_multiply(const Matrix matrix_a, const Matrix matrix_b) {
    size_t rows_a = matrix_a.rows;
    size_t cols_a = matrix_a.cols;
    size_t cols_b = matrix_b.cols;

    Matrix result_mat;
    
    if (matrix_a.cols != matrix_b.rows) {
        result_mat = seq_matrix_nulled(rows_a, cols_b);

        return result_mat;
    }

    result_mat = matrix_init(rows_a, cols_b);

    size_t matrix_a_size_bytes = rows_a * cols_a * sizeof(double);
    size_t matrix_b_size_bytes = cols_a * cols_b * sizeof(double);
    size_t result_size_bytes = rows_a * cols_b * sizeof(double);

    double *d_matrix_a_values;
    double *d_matrix_b_values;
    double *d_result;
    
    checkCuda( cudaMalloc(&d_matrix_a_values, matrix_a_size_bytes) );
    checkCuda( cudaMalloc(&d_matrix_b_values, matrix_b_size_bytes) );
    checkCuda( cudaMalloc(&d_result, result_size_bytes) );

    checkCuda( cudaMemcpy(d_matrix_a_values, matrix_a.values[0], matrix_a_size_bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_matrix_b_values, matrix_b.values[0], matrix_b_size_bytes, cudaMemcpyHostToDevice) );



    dim3 block(block_size, block_size);
    dim3 grid((cols_b+block.x-1)/block.x, (rows_a+block.y-1)/block.y);

    cuda_kernel_matrix_multiply<<<grid, block>>>(d_matrix_a_values, d_matrix_b_values, d_result, rows_a, cols_a, cols_b);
    
    checkCuda( cudaDeviceSynchronize() );

    checkCuda( cudaGetLastError() );

    checkCuda( cudaMemcpy(result_mat.values[0], d_result, result_size_bytes, cudaMemcpyDeviceToHost) );

    cudaFree(d_matrix_a_values);
    cudaFree(d_matrix_b_values);
    cudaFree(d_result);

    return result_mat;
}