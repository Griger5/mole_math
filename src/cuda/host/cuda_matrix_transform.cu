#include "../../../include/mole_math/cuda_matrix_transform.cuh"

#include "../device/cuda_kernel_matrix_transform.cuh"

void cuda_matrix_subtract_rows(Matrix *matrix, size_t row_minuend, size_t row_subtrahend, double multiplier) {
    int deviceId;
    int num_of_SM;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, deviceId);

    const int blocks_per_grid = 4 * num_of_SM;

    const size_t num_of_streams = 2;
    cudaStream_t streams[num_of_streams];

    for (size_t i = 0; i < num_of_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    size_t rows = matrix->rows;
    size_t cols = matrix->cols;
    size_t row_size_bytes = cols * sizeof(double);

    if (row_minuend >= rows || row_subtrahend >= rows || row_minuend == row_subtrahend) return;

    double *d_values_minuend, *d_values_subtrahend;
    
    cudaMalloc(&d_values_minuend, row_size_bytes);
    cudaMalloc(&d_values_subtrahend, row_size_bytes);

    size_t offset;

    for (size_t i = 0; i < num_of_streams; i++) {
        offset = (row_size_bytes/num_of_streams) * i;

        cudaMemcpyAsync(d_values_minuend + offset, matrix->values[row_minuend] + offset, row_size_bytes/num_of_streams, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_values_subtrahend + offset, matrix->values[row_subtrahend] + offset, row_size_bytes/num_of_streams, cudaMemcpyHostToDevice);

        cuda_kernel_matrix_subtract_rows<<<blocks_per_grid/num_of_streams, threads_per_block, 0, streams[i]>>>(d_values_minuend + offset, d_values_subtrahend + offset, multiplier, cols);

        cudaMemcpyAsync(matrix->values[row_minuend] + offset, d_values_minuend + offset, row_size_bytes/num_of_streams, cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();

    for (size_t i = 0; i < num_of_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_values_minuend);
    cudaFree(d_values_subtrahend);
}