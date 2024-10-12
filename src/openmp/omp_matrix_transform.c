#include <math.h>

#include "../../include/mole_math/omp_matrix_transform.h"
#include "../../include/mole_math/omp_matrix_properties.h"
#include "../../include/mole_math/omp_matrix_scalars.h"
#include "../../include/mole_math/omp_matrix_utils.h"
#include "../../include/mole_math/seq_matrix_utils.h"

void omp_matrix_subtract_rows(Matrix *matrix, size_t row_minuend, size_t row_subtrahend, double multiplier) {
    size_t cols = matrix->cols;
    
    if (row_minuend < matrix->rows && row_subtrahend < matrix->rows && row_minuend != row_subtrahend) {
        #pragma omp parallel for
        for (size_t i = 0; i < cols; i++) {
            matrix->values[row_minuend][i] = matrix->values[row_minuend][i] - multiplier * matrix->values[row_subtrahend][i];
        }
    }
}

Matrix omp_matrix_transpose(const Matrix matrix) {
    size_t rows = matrix.rows;
    size_t cols = matrix.cols;
    Matrix transposed;
    size_t j;

    if (matrix.values == NULL) return seq_matrix_nulled(rows, cols);

    if (rows == cols) {
        transposed = matrix_init(rows, cols);

        if (transposed.values != NULL) {
            #pragma omp parallel for private(j)
            for (size_t i = 0; i < rows; i++) {
                for (j = 0; j < cols; j++) {
                    transposed.values[i][j] = matrix.values[j][i];
                }
            }
        }
    }
    else {
        transposed = matrix_init(cols, rows);

        if (transposed.values != NULL) {
            #pragma omp parallel for private(j)
            for (size_t i = 0; i < cols; i++) {
                for (j = 0; j < rows; j++) {
                    transposed.values[i][j] = matrix.values[j][i];
                }
            }
        }
    }

    return transposed;
}

Matrix omp_matrix_ij_minor_matrix(const Matrix matrix, size_t i_row, size_t j_col) {
    size_t rows = matrix.rows;
    size_t cols = matrix.cols;

    if (i_row+1 <= 0 || j_col+1 <= 0) return seq_matrix_nulled(rows-1, cols-1);
    if (rows != cols || (i_row >= rows || j_col >= cols)) return seq_matrix_nulled(rows-1, cols-1);

    Matrix minor_matrix = matrix_init(rows-1, cols-1);

    size_t j;

    #pragma omp parallel
    {
        #pragma omp for private(j)
        for (size_t i = 0; i < i_row; i++) {
            for (j = 0; j < j_col; j++) {
                minor_matrix.values[i][j] = matrix.values[i][j];
            }
        }

        #pragma omp for private(j)
        for (size_t i = i_row; i < rows - 1; i++) {
            for (j = 0; j < j_col; j++) {
                minor_matrix.values[i][j] = matrix.values[i+1][j];
            }
        }

        #pragma omp for private(j)
        for (size_t i = 0; i < i_row; i++) {
            for (j = j_col; j < cols - 1; j++) {
                minor_matrix.values[i][j] = matrix.values[i][j+1];
            }
        }

        #pragma omp for private(j)
        for (size_t i = i_row; i < rows - 1; i++) {
            for (j = j_col; j < cols - 1; j++) {
                minor_matrix.values[i][j] = matrix.values[i+1][j+1];
            }
        }
    }

    return minor_matrix;
}

Matrix omp_matrix_inverse(Matrix matrix) {
    Matrix nulled = seq_matrix_nulled(matrix.rows, matrix.cols);
    
    if (matrix.rows != matrix.cols) return nulled;

    if (isinf(*matrix.determinant)) omp_matrix_determinant(matrix);

    if (*matrix.determinant == 0 || isnan(*matrix.determinant)) return nulled;

    size_t N = matrix.rows;
    Matrix inverted = omp_matrix_identity(N);

    Matrix matrix_copied = seq_matrix_copy(matrix);

    double ratio;

    for (size_t i = 0; i < N; i++) {
        if (matrix_copied.values[i][i] == 0) return nulled;

        #pragma omp parallel for private(ratio)
        for (size_t j = i+1; j < N; j++) {
            ratio = matrix_copied.values[j][i] / matrix_copied.values[i][i];
            omp_matrix_subtract_rows(&matrix_copied, j, i, ratio);
            omp_matrix_subtract_rows(&inverted, j, i, ratio);
        }
    }

    int j;

    #pragma omp parallel 
    {
        #pragma omp for private(j, ratio)
        for (int i = N-1; i >= 0; i--) {
            for (j = i; j >= 0; j--) {
                ratio = matrix_copied.values[j][i] / matrix_copied.values[i][i];
                omp_matrix_subtract_rows(&matrix_copied, j, i, ratio);
                omp_matrix_subtract_rows(&inverted, j, i, ratio);
            }
        }

        #pragma omp for
        for (size_t i = 0; i < N; i++) {
            omp_matrix_multiply_row_scalar(&inverted, i, 1/matrix_copied.values[i][i]);
        }
    }

    matrix_free(&matrix_copied);

    *inverted.determinant = 1/(*matrix.determinant);

    return inverted;
}