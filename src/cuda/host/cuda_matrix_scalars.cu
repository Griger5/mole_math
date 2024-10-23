#include "../../../include/mole_math/cuda_matrix_scalars.cuh"

#include "../device/cuda_kernel_matrix_scalars.cuh"

#include "../../../include/mole_math/cuda_check_error.cuh"

const int threads_per_block = 512;

void cuda_matrix_subtract_scalar(Matrix *matrix, double scalar) {
    if (matrix->values == NULL) return;
    
    int deviceId;
    int num_of_SM;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, deviceId);

    const int blocks_per_grid = 4 * num_of_SM;
    
    size_t rows = matrix->rows;
    size_t cols = matrix->cols;
    size_t matrix_size = rows * cols;
    size_t matrix_size_bytes = matrix_size * sizeof(double);

    double *d_matrix_values;
    
    checkCuda( cudaMalloc(&d_matrix_values, matrix_size_bytes) );

    checkCuda( cudaMemcpy(d_matrix_values, matrix->values[0], matrix_size_bytes, cudaMemcpyHostToDevice) );

    cuda_kernel_matrix_subtract_scalar<<<blocks_per_grid, threads_per_block>>>(d_matrix_values, scalar, matrix_size);
    
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaGetLastError() );

    checkCuda( cudaMemcpy(matrix->values[0], d_matrix_values, matrix_size_bytes, cudaMemcpyDeviceToHost) );

    cudaFree(d_matrix_values);
}

void cuda_matrix_multiply_row_scalar(Matrix *matrix, size_t row_num, double scalar) {
    if (matrix->values == NULL) return;
    
    int deviceId;
    int num_of_SM;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, deviceId);

    const int blocks_per_grid = 4 * num_of_SM;
    
    size_t cols = matrix->cols;
    size_t row_size_bytes = cols * sizeof(double);

    double *d_row_values;
    
    checkCuda( cudaMalloc(&d_row_values, row_size_bytes) );

    checkCuda( cudaMemcpy(d_row_values, matrix->values[row_num], row_size_bytes, cudaMemcpyHostToDevice) );

    cuda_kernel_matrix_multiply_row_scalar<<<blocks_per_grid, threads_per_block>>>(d_row_values, scalar, cols);
    
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaGetLastError() );

    checkCuda( cudaMemcpy(matrix->values[row_num], d_row_values, row_size_bytes, cudaMemcpyDeviceToHost) );

    cudaFree(d_row_values);
}