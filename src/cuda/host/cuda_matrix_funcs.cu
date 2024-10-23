#include "../../../include/mole_math/cuda_matrix_funcs.cuh"

#include "../device/cuda_kernel_matrix_funcs.cuh"

#include "../../../include/mole_math/cuda_check_error.cuh"

const int threads_per_block = 512;

double cuda_matrix_sum_row(const Matrix matrix, size_t row) {
    int deviceId;
    int num_of_SM;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, deviceId);

    const int blocks_per_grid = 4 * num_of_SM;
    
    size_t rows = matrix.rows;
    size_t cols = matrix.cols;
    size_t row_size_bytes = cols * sizeof(double);

    if (row >= rows) return NAN;

    double *d_row_values;
    double *d_sum_blocks, *d_sum;
    double h_sum;
    
    checkCuda( cudaMalloc(&d_row_values, row_size_bytes) );
    checkCuda( cudaMalloc(&d_sum_blocks, blocks_per_grid * sizeof(double)) );
    checkCuda( cudaMalloc(&d_sum, sizeof(double)) );

    checkCuda( cudaMemcpy(d_row_values, matrix.values[row], row_size_bytes, cudaMemcpyHostToDevice) );

    cuda_kernel_matrix_sum_row<<<blocks_per_grid, threads_per_block, threads_per_block*sizeof(double)>>>(d_row_values, cols, d_sum_blocks);
    checkCuda(cudaGetLastError());

    cuda_kernel_matrix_sum_row<<<1, 2*threads_per_block, 2*threads_per_block*sizeof(double)>>>(d_sum_blocks, blocks_per_grid, d_sum);
    checkCuda(cudaGetLastError());

    checkCuda( cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost) );

    cudaFree(d_row_values);
    cudaFree(d_sum_blocks);
    cudaFree(d_sum);

    return h_sum;
}