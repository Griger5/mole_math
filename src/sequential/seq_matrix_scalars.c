#include "../../include/mole_math/seq_matrix_scalars.h"

void seq_matrix_subtract_scalar(Matrix *matrix, double scalar) {
    size_t rows = matrix->rows;
    size_t cols = matrix->cols;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            matrix->values[i][j] = matrix->values[i][j] - scalar;
        }
    }
}

void seq_matrix_multiply_row_scalar(Matrix *matrix, size_t row_num, double scalar) {
    size_t cols = matrix->cols;
    
    if (row_num < matrix->rows) {
        for (size_t i = 0; i < cols; i++) {
            matrix->values[row_num][i] = matrix->values[row_num][i] * scalar;
        }
    }
}