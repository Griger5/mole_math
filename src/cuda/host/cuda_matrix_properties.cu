#include "../../../include/mole_math/cuda_matrix_properties.cuh"

#include "../device/cuda_kernel_matrix_properties.cuh"

#include "../../../include/mole_math/cuda_check_error.cuh"

#include "../../../include/mole_math/seq_matrix_transform.h"
#include "../../../include/mole_math/seq_matrix_utils.h"

double cuda_matrix_determinant(Matrix matrix) {
    if (matrix.rows != matrix.cols) return NAN;

    size_t N = matrix.rows;

    double determinant = 1;
    
    if (N == 1) {
        determinant = matrix.values[0][0];
        *matrix.determinant = determinant;

        return determinant;
    }

    if (N == 2) {
        determinant = (matrix.values[0][0] * matrix.values[1][1] - matrix.values[0][1] * matrix.values[1][0]);
        *matrix.determinant = determinant;

        return determinant;    
    }

    if (N == 3) {

        double aei = matrix.values[0][0] * matrix.values[1][1] * matrix.values[2][2];
        double bfg = matrix.values[0][1] * matrix.values[1][2] * matrix.values[2][0];
        double cdh = matrix.values[0][2] * matrix.values[1][0] * matrix.values[2][1];

        double ceg = matrix.values[0][2] * matrix.values[1][1] * matrix.values[2][0];
        double bdi = matrix.values[0][1] * matrix.values[1][0] * matrix.values[2][2];
        double afh = matrix.values[0][0] * matrix.values[1][2] * matrix.values[2][1];

        determinant = aei + bfg + cdh - ceg - bdi - afh;
        *matrix.determinant = determinant;

        return determinant;
    }

    int blocks_per_grid;
    
    size_t matrix_size_bytes = N * N * sizeof(double);

    Matrix copied_matrix = seq_matrix_copy(matrix);

    double *d_values;

    checkCuda( cudaMalloc(&d_values, matrix_size_bytes) );

    if (copied_matrix.values[0][0] == 0) {
        for (size_t i = 1; i < N; i++) {
            if (copied_matrix.values[i][i] != 0) {
                seq_matrix_switch_rows(&copied_matrix, 0, i);
                determinant *= -1;
                break;
            }       
        }
    }

    checkCuda ( cudaMemcpy(d_values, copied_matrix.values[0], matrix_size_bytes, cudaMemcpyHostToDevice) );

    int *managed_is_zero;
    
    checkCuda( cudaMallocManaged(&managed_is_zero, sizeof(int)) );
    *managed_is_zero = 0;

    for (size_t i = 0; i < N-1; i++) {
        cuda_kernel_matrix_determinant_check<<<1,1>>>(d_values, managed_is_zero, i, N);

        cudaDeviceSynchronize();

        if (*managed_is_zero) {
            *matrix.determinant = 0;
            return 0;
        }
        
        blocks_per_grid = N-i-1;

        cuda_kernel_matrix_determinant_zero_column<<<blocks_per_grid, threads_per_block>>>(d_values, N, i);
        checkCuda(cudaGetLastError());

        cudaDeviceSynchronize();
    }

    checkCuda( cudaMemcpy(copied_matrix.values[0], d_values, matrix_size_bytes, cudaMemcpyDeviceToHost) );

    for (size_t i = 0; i < N; i++) {
        determinant *= copied_matrix.values[i][i];
    }

    *matrix.determinant = determinant;

    matrix_free(&copied_matrix);

    cudaFree(d_values);
    cudaFree(managed_is_zero);

    return determinant;
}