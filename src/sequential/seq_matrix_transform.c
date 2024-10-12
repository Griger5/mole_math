#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/mole_math/seq_matrix_transform.h"
#include "../../include/mole_math/seq_matrix_properties.h"
#include "../../include/mole_math/seq_matrix_scalars.h"
#include "../../include/mole_math/seq_matrix_utils.h"

void seq_matrix_subtract_rows(Matrix *matrix, size_t row_minuend, size_t row_subtrahend, double multiplier) {
    if (row_minuend < matrix->rows && row_subtrahend < matrix->rows && row_minuend != row_subtrahend) {
        for (size_t i = 0; i < matrix->cols; i++) {
            matrix->values[row_minuend][i] = matrix->values[row_minuend][i] - multiplier * matrix->values[row_subtrahend][i];
        }
    }
}

void seq_matrix_switch_rows(Matrix *matrix, size_t row_1, size_t row_2) {
    size_t rows = matrix->rows;
    size_t cols = matrix->cols;
    size_t row_size_bytes = cols * sizeof(double);

    if (row_1 >= rows || row_2 >= rows) return;

    double *temp = malloc(row_size_bytes);

    memcpy(temp, matrix->values[row_1], row_size_bytes);
    memcpy(matrix->values[row_1], matrix->values[row_2], row_size_bytes);
    memcpy(matrix->values[row_2], temp, row_size_bytes);

    free(temp);

    if (matrix->determinant != NULL) {
        if (!isinf(*matrix->determinant)) *matrix->determinant *= -1;
    }
}

Matrix seq_matrix_transpose(const Matrix matrix) {
    size_t rows = matrix.rows;
    size_t cols = matrix.cols;
    Matrix transposed;

    if (matrix.values == NULL) return seq_matrix_nulled(rows, cols);

    if (rows == cols) {
        transposed = matrix_init(rows, cols);

        if (transposed.values != NULL) {
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    transposed.values[i][j] = matrix.values[j][i];
                }
            }
        }
    }
    else {
        transposed = matrix_init(cols, rows);

        if (transposed.values != NULL) {
            for (size_t i = 0; i < cols; i++) {
                for (size_t j = 0; j < rows; j++) {
                    transposed.values[i][j] = matrix.values[j][i];
                }
            }
        }
    }

    return transposed;
}

Matrix seq_matrix_ij_minor_matrix(const Matrix matrix, size_t i_row, size_t j_col) {
    size_t rows = matrix.rows;
    size_t cols = matrix.cols;

    if (i_row+1 <= 0 || j_col+1 <= 0) return seq_matrix_nulled(rows-1, cols-1);
    if (rows != cols || (i_row >= rows || j_col >= cols)) return seq_matrix_nulled(rows-1, cols-1);

    Matrix minor_matrix = matrix_init(rows-1, cols-1);

    for (size_t i = 0; i < i_row; i++) {
        for (size_t j = 0; j < j_col; j++) {
            minor_matrix.values[i][j] = matrix.values[i][j];
        }
    }

    for (size_t i = i_row; i < rows - 1; i++) {
        for (size_t j = 0; j < j_col; j++) {
            minor_matrix.values[i][j] = matrix.values[i+1][j];
        }
    }

    for (size_t i = 0; i < i_row; i++) {
        for (size_t j = j_col; j < cols - 1; j++) {
            minor_matrix.values[i][j] = matrix.values[i][j+1];
        }
    }

    for (size_t i = i_row; i < rows - 1; i++) {
        for (size_t j = j_col; j < cols - 1; j++) {
            minor_matrix.values[i][j] = matrix.values[i+1][j+1];
        }
    }

    return minor_matrix;
}

Matrix seq_matrix_inverse(Matrix matrix) {
    Matrix nulled = seq_matrix_nulled(matrix.rows, matrix.cols);
    
    if (matrix.rows != matrix.cols) return nulled;

    if (matrix.values == NULL || matrix.determinant == NULL) return nulled;

    if (isinf(*matrix.determinant)) seq_matrix_determinant(matrix);

    if (*matrix.determinant == 0 || isnan(*matrix.determinant)) return nulled;

    size_t N = matrix.rows;
    Matrix inverted = seq_matrix_identity(N);

    Matrix matrix_copied = seq_matrix_copy(matrix);

    double ratio;

    for (size_t i = 0; i < N; i++) {
        if (matrix_copied.values[i][i] == 0) return nulled;

        for (size_t j = i+1; j < N; j++) {
            ratio = matrix_copied.values[j][i] / matrix_copied.values[i][i];
            seq_matrix_subtract_rows(&matrix_copied, j, i, ratio);
            seq_matrix_subtract_rows(&inverted, j, i, ratio);
        }
    }

    for (int i = N-1; i >= 0; i--) {
        for (int j = i; j >= 0; j--) {
            ratio = matrix_copied.values[j][i] / matrix_copied.values[i][i];
            seq_matrix_subtract_rows(&matrix_copied, j, i, ratio);
            seq_matrix_subtract_rows(&inverted, j, i, ratio);
        }
    }

    for (size_t i = 0; i < N; i++) {
        seq_matrix_multiply_row_scalar(&inverted, i, 1/matrix_copied.values[i][i]);
    }

    matrix_free(&matrix_copied);

    *inverted.determinant = 1/(*matrix.determinant);

    return inverted;
}