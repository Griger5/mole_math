#include <stdio.h>
#include <omp.h>

#include "../../../include/mole_math/matrix_utils.h"

#include "../../../include/mole_math/seq_matrix_utils.h"
#include "../../../include/mole_math/omp_matrix_utils.h"

void matrix_print(const Matrix matrix) {
    size_t rows = matrix.rows;
    size_t cols = matrix.cols;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            printf("%f ", matrix.values[i][j]);
        }
        printf("\n");
    }
}

Matrix matrix_identity(size_t N, char flag) {
    size_t problem_size = N;

    Matrix result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_identity(N);
            break;
        case 's':
            result = seq_matrix_identity(N);
            break;
        default:
            if ((double)problem_size/omp_get_num_procs() >= 150/16.0) result = omp_matrix_identity(N);
            else result = seq_matrix_identity(N);
            break;
    }

    return result;
}

Matrix matrix_nulled(size_t rows, size_t cols) {
    return seq_matrix_nulled(rows, cols);
}

Matrix matrix_random(size_t rows, size_t cols, char flag) {
    Matrix result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_random(rows, cols);
            break;
        case 's':
            result = seq_matrix_random(rows, cols);
            break;
        default:
            result = seq_matrix_random(rows, cols);
            break;
    }

    return result;
}

Matrix matrix_init_integers(size_t rows, size_t cols, char flag) {
    size_t problem_size = rows * cols;

    Matrix result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_init_integers(rows, cols);
            break;
        case 's':
            result = seq_matrix_init_integers(rows, cols);
            break;
        default:
            if ((double)problem_size/omp_get_num_procs() >= 2500/16.0) result = omp_matrix_init_integers(rows, cols);
            else result = seq_matrix_init_integers(rows, cols);
            break;
    }

    return result;
}

Matrix matrix_copy(const Matrix matrix_to_copy) {
    return seq_matrix_copy(matrix_to_copy);
}

void matrix_replace(Matrix *to_replace, const Matrix matrix_to_copy, char flag) {
    size_t problem_size = matrix_to_copy.rows * matrix_to_copy.cols;
    
    switch (flag) {
        case 'o':
            omp_matrix_replace(to_replace, matrix_to_copy);
            break;
        case 's':
            seq_matrix_replace(to_replace, matrix_to_copy);
            break;
        default:
            if ((double)problem_size/omp_get_num_procs() >= 4096/16.0) omp_matrix_replace(to_replace, matrix_to_copy);
            else seq_matrix_replace(to_replace, matrix_to_copy);
            break;
    }
}

Matrix matrix_array_to_matrix(double *array, size_t length) {
    return seq_matrix_array_to_matrix(array, length);
}