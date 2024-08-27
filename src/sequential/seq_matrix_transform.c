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
    size_t cols = matrix->cols;
    double temp;

    if (row_1 < matrix->rows && row_2 < matrix->rows) {
        for (size_t i = 0; i < cols; i++) {
            temp = matrix->values[row_1][i];
            matrix->values[row_1][i] = matrix->values[row_2][i];
            matrix->values[row_2][i] = temp;        
        }
    }
}

Matrix seq_matrix_inverse(const Matrix matrix) {
    Matrix nulled = seq_matrix_nulled(matrix.rows, matrix.cols);
    
    if (matrix.rows != matrix.cols || seq_matrix_determinant(matrix) == 0) return nulled;

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

    return inverted;
}