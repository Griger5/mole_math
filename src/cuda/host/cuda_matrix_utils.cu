#include "../../../include/mole_math/cuda_matrix_utils.cuh"

#include "../device/cuda_kernel_matrix_utils.cuh"

#include "../../../include/mole_math/cuda_check_error.cuh"

const int threads_per_block = 512;

Matrix cuda_matrix_identity(size_t N) {
    Matrix identity = matrix_init(N, N);

    if (identity.values == NULL) return identity;

    int deviceId;
    int num_of_SM;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, deviceId);
    
    const int blocks_per_grid = 4 * num_of_SM;
    
    size_t matrix_size_bytes = N * N * sizeof(double);
    
    double *d_values;
    
    checkCuda( cudaMalloc(&d_values, matrix_size_bytes) );

    checkCuda( cudaMemset(d_values, 0, matrix_size_bytes) );

    cuda_kernel_matrix_identity<<<blocks_per_grid, threads_per_block>>>(d_values, N);

    checkCuda( cudaDeviceSynchronize() );

    checkCuda( cudaMemcpy(identity.values[0], d_values, matrix_size_bytes, cudaMemcpyDeviceToHost) );

    cudaFree(d_values);

    return identity;
}

Matrix cuda_matrix_random(size_t rows, size_t cols) {
    Matrix random = matrix_init(rows, cols);

    if (random.values == NULL) return random;

    int deviceId;
    int num_of_SM;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, deviceId);
    
    const int blocks_per_grid = 4 * num_of_SM;
    
    size_t matrix_size = rows * cols;
    size_t matrix_size_bytes = matrix_size * sizeof(double);
    
    double *d_values;
    
    checkCuda( cudaMalloc(&d_values, matrix_size_bytes) );

    cuda_kernel_matrix_random<<<blocks_per_grid, threads_per_block>>>(d_values, matrix_size);

    checkCuda( cudaDeviceSynchronize() );

    checkCuda( cudaMemcpy(random.values[0], d_values, matrix_size_bytes, cudaMemcpyDeviceToHost) );

    cudaFree(d_values);

    return random;
}

Matrix cuda_matrix_init_integers(size_t rows, size_t cols) {
    Matrix integers = matrix_init(rows, cols);

    if (integers.values == NULL) return integers;

    int deviceId;
    int num_of_SM;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, deviceId);
    
    const int blocks_per_grid = 4 * num_of_SM;
    
    size_t matrix_size = rows * cols;
    size_t matrix_size_bytes = matrix_size * sizeof(double);
    
    double *d_values;
    
    checkCuda( cudaMalloc(&d_values, matrix_size_bytes) );

    cuda_kernel_matrix_init_integers<<<blocks_per_grid, threads_per_block>>>(d_values, matrix_size);

    checkCuda( cudaDeviceSynchronize() );

    checkCuda( cudaMemcpy(integers.values[0], d_values, matrix_size_bytes, cudaMemcpyDeviceToHost) );

    cudaFree(d_values);

    return integers;
}