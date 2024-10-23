#include "../../../include/mole_math/cuda_matrix_operations.cuh"

#include "../device/cuda_kernel_matrix_operations.cuh"

#include "../../../include/mole_math/cuda_check_error.cuh"

#include "../../../include/mole_math/seq_matrix_utils.h"

const int threads_per_block = 512;
const int block_size = 32;

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

    if (result_mat.values == NULL) return result_mat;

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

Matrix cuda_matrix_subtract_elements(const Matrix matrix_a, const Matrix matrix_b) {
    size_t rows = matrix_a.rows;
    size_t cols = matrix_a.cols;

    Matrix result_mat;
    
    if (matrix_a.rows != matrix_b.rows || matrix_a.cols != matrix_b.cols) {
        result_mat = seq_matrix_nulled(rows, cols);

        return result_mat;
    }

    result_mat = matrix_init(rows, cols);

    if (result_mat.values == NULL) return result_mat;

    int deviceId;
    int num_of_SM;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, deviceId);

    const int blocks_per_grid = 8 * num_of_SM;

    size_t matrix_size = rows * cols;
    size_t matrix_size_bytes = matrix_size * sizeof(double);

    double *d_matrix_a_values;
    double *d_matrix_b_values;
    double *d_result;
    
    checkCuda( cudaMalloc(&d_matrix_a_values, matrix_size_bytes) );
    checkCuda( cudaMalloc(&d_matrix_b_values, matrix_size_bytes) );
    checkCuda( cudaMalloc(&d_result, matrix_size_bytes) );

    checkCuda( cudaMemcpy(d_matrix_a_values, matrix_a.values[0], matrix_size_bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_matrix_b_values, matrix_b.values[0], matrix_size_bytes, cudaMemcpyHostToDevice) );

    cuda_kernel_matrix_subtract_elements<<<blocks_per_grid, threads_per_block>>>(d_matrix_a_values, d_matrix_b_values, d_result, rows, cols);
    
    checkCuda( cudaDeviceSynchronize() );

    checkCuda( cudaGetLastError() );

    checkCuda( cudaMemcpy(result_mat.values[0], d_result, matrix_size_bytes, cudaMemcpyDeviceToHost) );

    cudaFree(d_matrix_a_values);
    cudaFree(d_matrix_b_values);
    cudaFree(d_result);

    return result_mat;
}

Matrix cuda_matrix_multiply_elements(const Matrix matrix_a, const Matrix matrix_b) {
    size_t rows = matrix_a.rows;
    size_t cols = matrix_a.cols;

    Matrix result_mat;
    
    if (matrix_a.rows != matrix_b.rows || matrix_a.cols != matrix_b.cols) {
        result_mat = seq_matrix_nulled(rows, cols);

        return result_mat;
    }

    result_mat = matrix_init(rows, cols);

    if (result_mat.values == NULL) return result_mat;

    int deviceId;
    int num_of_SM;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, deviceId);

    const int blocks_per_grid = 8 * num_of_SM;

    size_t matrix_size = rows * cols;
    size_t matrix_size_bytes = matrix_size * sizeof(double);

    double *d_matrix_a_values;
    double *d_matrix_b_values;
    double *d_result;
    
    checkCuda( cudaMalloc(&d_matrix_a_values, matrix_size_bytes) );
    checkCuda( cudaMalloc(&d_matrix_b_values, matrix_size_bytes) );
    checkCuda( cudaMalloc(&d_result, matrix_size_bytes) );

    checkCuda( cudaMemcpy(d_matrix_a_values, matrix_a.values[0], matrix_size_bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_matrix_b_values, matrix_b.values[0], matrix_size_bytes, cudaMemcpyHostToDevice) );

    cuda_kernel_matrix_multiply_elements<<<blocks_per_grid, threads_per_block>>>(d_matrix_a_values, d_matrix_b_values, d_result, rows, cols);
    
    checkCuda( cudaDeviceSynchronize() );

    checkCuda( cudaGetLastError() );

    checkCuda( cudaMemcpy(result_mat.values[0], d_result, matrix_size_bytes, cudaMemcpyDeviceToHost) );

    cudaFree(d_matrix_a_values);
    cudaFree(d_matrix_b_values);
    cudaFree(d_result);

    return result_mat;
}