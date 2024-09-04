#include <stdio.h>

#include "../../../include/mole_math/matrix_utils.h"

#include "../../../include/mole_math/seq_matrix_utils.h"

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

Matrix matrix_identity(size_t N) {
    return seq_matrix_identity(N);
}

Matrix matrix_nulled(size_t rows, size_t cols) {
    return seq_matrix_nulled(rows, cols);
}

Matrix matrix_random(size_t rows, size_t cols) {
    return seq_matrix_random(rows, cols);
}

Matrix matrix_copy(const Matrix matrix_to_copy) {
    return seq_matrix_copy(matrix_to_copy);
}

void matrix_replace(Matrix *to_replace, const Matrix matrix_to_copy) {
    seq_matrix_replace(to_replace, matrix_to_copy);
}

Matrix matrix_array_to_matrix(double *array, size_t length) {
    return seq_matrix_array_to_matrix(array, length);
}