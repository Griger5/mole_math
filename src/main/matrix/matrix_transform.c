#include <omp.h>

#include "../../../include/mole_math/matrix_transform.h"

#include "../../../include/mole_math/seq_matrix_transform.h"
#include "../../../include/mole_math/omp_matrix_transform.h"
#include "../../../include/mole_math/cuda_matrix_transform.cuh"

void matrix_switch_rows(Matrix *matrix, size_t row_1, size_t row_2) {
    seq_matrix_switch_rows(matrix, row_1, row_2);
}

void matrix_subtract_rows(Matrix *matrix, size_t row_minuend, size_t row_subtrahend, double multiplier, char flag) {
    size_t problem_size = matrix->cols;
    
    switch (flag) {
        case 'o':
            omp_matrix_subtract_rows(matrix, row_minuend, row_subtrahend, multiplier);
            break;
        case 's':
            seq_matrix_subtract_rows(matrix, row_minuend, row_subtrahend, multiplier);
            break;
        #ifdef CUDA_SM_COUNT
        case 'c':
            cuda_matrix_subtract_rows(matrix, row_minuend, row_subtrahend, multiplier);
            break;
        #endif
        default:
            if ((double)problem_size/omp_get_num_procs() >= 2000/16.0) omp_matrix_subtract_rows(matrix, row_minuend, row_subtrahend, multiplier);
            else seq_matrix_subtract_rows(matrix, row_minuend, row_subtrahend, multiplier);
            break;
    }
}

Matrix matrix_transpose(const Matrix matrix, char flag) {
    size_t problem_size = matrix.rows * matrix.cols;

    Matrix result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_transpose(matrix);
            break;
        case 's':
            result = seq_matrix_transpose(matrix);
            break;
        #ifdef CUDA_SM_COUNT
        case 'c':
            result = cuda_matrix_transpose(matrix);
            break;
        #endif
        default:
            if ((double)problem_size/omp_get_num_procs() >= 16384/16.0) result = omp_matrix_transpose(matrix);
            else result = seq_matrix_transpose(matrix);
            break;
    }

    return result;
}

Matrix matrix_ij_minor_matrix(const Matrix matrix, size_t i_row, size_t j_col, char flag) {
    size_t problem_size = matrix.rows * matrix.cols;

    Matrix result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_ij_minor_matrix(matrix, i_row, j_col);
            break;
        case 's':
            result = seq_matrix_ij_minor_matrix(matrix, i_row, j_col);
            break;
        default:
            if ((double)problem_size/omp_get_num_procs() >= 16384/16.0) result = omp_matrix_ij_minor_matrix(matrix, i_row, j_col);
            else result = seq_matrix_ij_minor_matrix(matrix, i_row, j_col);
            break;
    }

    return result;
}

Matrix matrix_inverse(Matrix matrix, char flag) {
    size_t problem_size = matrix.rows * matrix.cols;

    Matrix result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_inverse(matrix);
            break;
        case 's':
            result = seq_matrix_inverse(matrix);
            break;
        default:
            if ((double)problem_size/omp_get_num_procs() >= 2500/16.0) result = omp_matrix_inverse(matrix);
            else result = seq_matrix_inverse(matrix);
            break;
    }

    return result;
}
