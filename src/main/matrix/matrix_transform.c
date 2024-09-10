#include <omp.h>

#include "../../../include/mole_math/matrix_transform.h"

#include "../../../include/mole_math/seq_matrix_transform.h"
#include "../../../include/mole_math/omp_matrix_transform.h"

void matrix_switch_rows(Matrix *matrix, size_t row_1, size_t row_2, char flag) {
    size_t problem_size = matrix->cols;
    
    switch (flag) {
        case 'o':
            omp_matrix_switch_rows(matrix, row_1, row_2);
            break;
        case 's':
            seq_matrix_switch_rows(matrix, row_1, row_2);
            break;
        default:
            if ((double)problem_size/omp_get_num_procs() >= 2000/16.0) omp_matrix_switch_rows(matrix, row_1, row_2);
            else seq_matrix_switch_rows(matrix, row_1, row_2);
            break;
    }
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
        default:
            if ((double)problem_size/omp_get_num_procs() >= 2000/16.0) omp_matrix_subtract_rows(matrix, row_minuend, row_subtrahend, multiplier);
            else seq_matrix_subtract_rows(matrix, row_minuend, row_subtrahend, multiplier);
            break;
    }
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
