#include <omp.h>

#include "../../../include/mole_math/matrix_scalars.h"

#include "../../../include/mole_math/seq_matrix_scalars.h"
#include "../../../include/mole_math/omp_matrix_scalars.h"
#include "../../../include/mole_math/cuda_matrix_scalars.cuh"

void matrix_subtract_scalar(Matrix *matrix, double scalar, char flag) {
    size_t problem_size = matrix->rows * matrix->cols;
    
    switch (flag) {
        case 'o':
            omp_matrix_subtract_scalar(matrix, scalar);
            break;
        case 's':
            seq_matrix_subtract_scalar(matrix, scalar);
            break;
        #ifdef CUDA_SM_COUNT
        case 'c':
            cuda_matrix_subtract_scalar(matrix, scalar);
            break;
        #endif
        default:
            if ((double)problem_size/omp_get_num_procs() >= 10000/16.0) omp_matrix_subtract_scalar(matrix, scalar);
            else seq_matrix_subtract_scalar(matrix, scalar);
            break;
    }
}

void matrix_multiply_row_scalar(Matrix *matrix, size_t row_num, double scalar, char flag) {
    size_t problem_size = matrix->cols;
    
    switch (flag) {
        case 'o':
            omp_matrix_multiply_row_scalar(matrix, row_num, scalar);
            break;
        case 's':
            seq_matrix_multiply_row_scalar(matrix, row_num, scalar);
            break;
        #ifdef CUDA_SM_COUNT
        case 'c':
            cuda_matrix_multiply_row_scalar(matrix, row_num, scalar);
            break;
        #endif
        default:
            if ((double)problem_size/omp_get_num_procs() >= 2000/16.0) omp_matrix_multiply_row_scalar(matrix, row_num, scalar);
            else seq_matrix_multiply_row_scalar(matrix, row_num, scalar);
            break;
    }
}