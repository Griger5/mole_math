#include <omp.h>

#include "../../../include/mole_math/matrix_operations.h"

#include "../../../include/mole_math/seq_matrix_operations.h"
#include "../../../include/mole_math/omp_matrix_operations.h"


Matrix matrix_multiply(const Matrix matrix_a, const Matrix matrix_b, char flag) {
    size_t problem_size = matrix_a.rows * matrix_a.cols * matrix_b.cols;

    Matrix result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_multiply(matrix_a, matrix_b);
            break;
        case 's':
            result = seq_matrix_multiply(matrix_b, matrix_b);
            break;
        default:
            if ((double)problem_size/omp_get_num_procs() >= 125000/16.0) result = omp_matrix_multiply(matrix_a, matrix_b);
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
            result = seq_matrix_subtract_elements(matrix_b, matrix_b);
            break;
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
            result = seq_matrix_multiply_elements(matrix_b, matrix_b);
            break;
        default:
            if ((double)problem_size/omp_get_num_procs() >= 65536/16.0) result = omp_matrix_multiply_elements(matrix_a, matrix_b);
            else result = seq_matrix_multiply_elements(matrix_a, matrix_b);
            break;
    }

    return result;
}