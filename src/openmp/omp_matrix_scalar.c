#include "../../include/mole_math/omp_matrix_scalars.h"

void omp_matrix_subtract_scalar(Matrix *matrix, double scalar) {
    size_t rows = matrix->rows;
    size_t cols = matrix->cols;

    size_t i, j;

    #pragma omp parallel for private(i,j) shared(matrix, scalar)
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            matrix->values[i][j] = matrix->values[i][j] - scalar;
        }
    }
}

void omp_matrix_multiply_row_scalar(Matrix *matrix, size_t row_num, double scalar) {
    size_t cols = matrix->cols;
    
    size_t i;

    if (row_num < matrix->rows) {
        #pragma omp parallel for private(i) shared(matrix, scalar)
        for (i = 0; i < cols; i++) {
            matrix->values[row_num][i] = matrix->values[row_num][i] * scalar;
        }
    }
}