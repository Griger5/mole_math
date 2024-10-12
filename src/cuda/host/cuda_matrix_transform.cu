#include "../../../include/mole_math/cuda_matrix_transform.cuh"

#include "../device/cuda_kernel_matrix_transform.cuh"

#include "../../../include/mole_math/cuda_check_error.cuh"

#include "../../../include/mole_math/seq_matrix_utils.h"

void cuda_matrix_subtract_rows(Matrix *matrix, size_t row_minuend, size_t row_subtrahend, double multiplier) {
    int deviceId;
    int num_of_SM;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, deviceId);
    
    const int blocks_per_grid = 4 * num_of_SM;
    
    size_t rows = matrix->rows;
    size_t cols = matrix->cols;
    size_t row_size_bytes = cols * sizeof(double);

    if (row_minuend >= rows || row_subtrahend >= rows || row_minuend == row_subtrahend) return;
    
    double *d_values_minuend, *d_values_subtrahend;
    
    checkCuda( cudaMalloc(&d_values_minuend, row_size_bytes) );
    checkCuda( cudaMalloc(&d_values_subtrahend, row_size_bytes) );

    checkCuda( cudaMemcpy(d_values_minuend, matrix->values[row_minuend], row_size_bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_values_subtrahend, matrix->values[row_subtrahend], row_size_bytes, cudaMemcpyHostToDevice) );

    cuda_kernel_matrix_subtract_rows<<<blocks_per_grid, threads_per_block>>>(d_values_minuend, d_values_subtrahend, multiplier, cols);

    checkCuda( cudaDeviceSynchronize() );

    checkCuda( cudaMemcpy(matrix->values[row_minuend], d_values_minuend, row_size_bytes, cudaMemcpyDeviceToHost) );

    cudaFree(d_values_minuend);
    cudaFree(d_values_subtrahend);
}

Matrix cuda_matrix_transpose(const Matrix matrix) {
    size_t rows = matrix.rows;
    size_t cols = matrix.cols;
    size_t matrix_size_bytes = rows * cols * sizeof(double);

    Matrix transposed;

    if (matrix.values == NULL) return seq_matrix_nulled(rows, cols);

    double *d_matrix_values;
    double *d_transposed_values;

    checkCuda( cudaMalloc(&d_matrix_values, matrix_size_bytes) );
    checkCuda( cudaMalloc(&d_transposed_values, matrix_size_bytes) );

    cudaMemcpy(d_matrix_values, matrix.values[0], matrix_size_bytes, cudaMemcpyHostToDevice);

    dim3 block(block_size, block_size);
    dim3 grid((cols+block.x-1)/block.x, (rows+block.y-1)/block.y);

    if (rows == cols) {
        transposed = matrix_init(rows, cols);
    
        if (transposed.values != NULL) {
            cuda_kernel_matrix_transpose<<<grid, block>>>(d_matrix_values, d_transposed_values, rows, cols);
        }
    }
    else {
        transposed = matrix_init(cols, rows);
        
        if (transposed.values != NULL) {
            cuda_kernel_matrix_transpose_flip<<<grid, block>>>(d_matrix_values, d_transposed_values, rows, cols);
        }
    }

    checkCuda( cudaDeviceSynchronize() );

    checkCuda( cudaGetLastError() );

    checkCuda( cudaMemcpy(transposed.values[0], d_transposed_values, matrix_size_bytes, cudaMemcpyDeviceToHost) );

    if (matrix.determinant != NULL) {
        if (!isinf(*matrix.determinant)) *transposed.determinant = *matrix.determinant;
    }

    cudaFree(d_matrix_values);
    cudaFree(d_transposed_values);

    return transposed;
}