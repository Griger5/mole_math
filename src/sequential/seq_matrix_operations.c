#include <math.h>

#include "../../include/mole_math/seq_matrix_operations.h"
#include "../../include/mole_math/seq_matrix_utils.h"

Matrix seq_matrix_multiply(const Matrix matrix_a, const Matrix matrix_b) {
    size_t rows_a = matrix_a.rows;
    size_t cols_a = matrix_a.cols;
    size_t cols_b = matrix_b.cols;

    Matrix result;
    
    if (matrix_a.cols != matrix_b.rows) {
        result = seq_matrix_nulled(rows_a, cols_b);

        return result;
    }

    result = matrix_init(rows_a, cols_b);

    if (result.values != NULL) {
        for(size_t row = 0; row < rows_a; row++) {
            for(size_t col = 0; col < cols_b; col++) {
                for (size_t k = 0; k < cols_a; k++) {
                    result.values[row][col] += matrix_a.values[row][k] * matrix_b.values[k][col];
                }
            }
        }
    }

    if (matrix_a.determinant != NULL && matrix_b.determinant != NULL) {
        if (!isinf(*matrix_a.determinant) && !isinf(*matrix_b.determinant)) {
            *result.determinant = (*matrix_a.determinant) * (*matrix_b.determinant);
        }
    }

    return result;
}

Matrix seq_matrix_subtract_elements(const Matrix matrix_a, const Matrix matrix_b) {
    size_t rows = matrix_a.rows;
    size_t cols = matrix_a.cols;

    Matrix result;
    
    if (matrix_a.rows != matrix_b.rows || matrix_a.cols != matrix_b.cols) {
        result = seq_matrix_nulled(rows, cols);

        return result;
    }

    result = matrix_init(rows, cols);

    if (result.values != NULL) {
        for(size_t row = 0; row < rows; row++) {
            for(size_t col = 0; col < cols; col++) {
                result.values[row][col] = matrix_a.values[row][col] - matrix_b.values[row][col];
            }
        }
    }

    return result;
}

Matrix seq_matrix_multiply_elements(const Matrix matrix_a, const Matrix matrix_b) {
    size_t rows = matrix_a.rows;
    size_t cols = matrix_a.cols;

    Matrix result;

    if (matrix_a.rows != matrix_b.rows || matrix_a.cols != matrix_b.cols) {
        result = seq_matrix_nulled(rows, cols);

        return result;
    }

    result = matrix_init(rows, cols);

    if (result.values != NULL) {
        for(size_t row = 0; row < rows; row++) {
            for(size_t col = 0; col < cols; col++) {
                result.values[row][col] = matrix_a.values[row][col] * matrix_b.values[row][col];
            }
        }
    }

    return result;
}