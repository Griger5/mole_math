#include <omp.h>

#include "../../../include/mole_math/matrix_operations.h"

#include "../../../include/mole_math/seq_matrix_operations.h"
#include "../../../include/mole_math/omp_matrix_operations.h"
#include "../../../include/mole_math/cuda_matrix_operations.cuh"

Matrix matrix_multiply(const Matrix matrix_a, const Matrix matrix_b, char flag) {
    size_t problem_size = matrix_a.rows * matrix_a.cols * matrix_b.cols;

    Matrix result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_multiply(matrix_a, matrix_b);
            break;
        case 's':
            result = seq_matrix_multiply(matrix_a, matrix_b);
            break;
        #ifdef CUDA_SM_COUNT
        case 'c':
            result = cuda_matrix_multiply(matrix_a, matrix_b);
            break;
        #endif
        default:
            if ((double)problem_size/omp_get_num_procs() >= 125000/16.0) {
                #ifdef CUDA_SM_COUNT
                    if ((double)omp_get_num_procs() * problem_size / CUDA_SM_COUNT >= 4 * 262144 / 5) result = cuda_matrix_multiply(matrix_a, matrix_b);
                    else result = omp_matrix_multiply(matrix_a, matrix_b);
                #else
                    result = omp_matrix_multiply(matrix_a, matrix_b);
                #endif
            } 
            else result = seq_matrix_multiply(matrix_a, matrix_b);
            break;
    }

    return result;
}

Matrix matrix_subtract_elements(const Matrix matrix_a, const Matrix matrix_b, char flag) {
    size_t problem_size = matrix_a.rows * matrix_a.cols;

    Matrix result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_subtract_elements(matrix_a, matrix_b);
            break;
        case 's':
            result = seq_matrix_subtract_elements(matrix_a, matrix_b);
            break;
        #ifdef CUDA_SM_COUNT
        case 'c':
            result = cuda_matrix_subtract_elements(matrix_a, matrix_b);
            break;
        #endif
        default:
            if ((double)problem_size/omp_get_num_procs() >= 65536/16.0) result = omp_matrix_subtract_elements(matrix_a, matrix_b);
            else result = seq_matrix_subtract_elements(matrix_a, matrix_b);
            break;
    }

    return result;
}

Matrix matrix_multiply_elements(const Matrix matrix_a, const Matrix matrix_b, char flag) {
    size_t problem_size = matrix_a.rows * matrix_a.cols;

    Matrix result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_multiply_elements(matrix_a, matrix_b);
            break;
        case 's':
            result = seq_matrix_multiply_elements(matrix_a, matrix_b);
            break;
        #ifdef CUDA_SM_COUNT
        case 'c':
            result = cuda_matrix_multiply_elements(matrix_a, matrix_b);
            break;
        #endif
        default:
            if ((double)problem_size/omp_get_num_procs() >= 65536/16.0) result = omp_matrix_multiply_elements(matrix_a, matrix_b);
            else result = seq_matrix_multiply_elements(matrix_a, matrix_b);
            break;
    }

    return result;
}