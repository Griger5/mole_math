#include "../../../include/mole_math/cuda_matrix_transform.cuh"

#include "../device/cuda_kernel_matrix_transform.cuh"

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

    float *d_values_minuend, *d_values_subtrahend;
    
    cudaMalloc(&d_values_minuend, row_size_bytes);
    cudaMalloc(&d_values_subtrahend, row_size_bytes);

    cudaMemcpy(d_values_minuend, matrix->values[row_minuend], row_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_subtrahend, matrix->values[row_subtrahend], row_size_bytes, cudaMemcpyHostToDevice);

    cuda_kernel_matrix_subtract_rows<<<blocks_per_grid, threads_per_block>>>(d_values_minuend, d_values_subtrahend, multiplier, cols);

    cudaMemcpy(matrix->values[row_minuend], d_values_minuend, row_size_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_values_minuend);
    cudaFree(d_values_subtrahend);
}