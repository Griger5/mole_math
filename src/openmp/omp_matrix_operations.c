#include "../../include/mole_math/omp_matrix_operations.h"
#include "../../include/mole_math/omp_matrix_utils.h"

Matrix omp_matrix_multiply(const Matrix matrix_a, const Matrix matrix_b) {
    size_t rows_a = matrix_a.rows;
    size_t cols_a = matrix_a.cols;
    size_t cols_b = matrix_b.cols;

    Matrix result;
    
    if (matrix_a.cols != matrix_b.rows) {
        result = omp_matrix_nulled(rows_a, cols_b);

        return result;
    }

    result = matrix_init(rows_a, cols_b);

    size_t row, col, k;

    if (result.values != NULL) {
        #pragma omp parallel private(row, col, k) shared(result, matrix_a, matrix_b)
        {
            #pragma omp for
            for(row = 0; row < rows_a; row++) {
                for (k = 0; k < cols_a; k++) {
                    for (col = 0; col < cols_b; col++) {
                        result.values[row][col] += matrix_a.values[row][k] * matrix_b.values[k][col];
                    }
                }
            }
        }
    }

    return result;
}

Matrix omp_matrix_subtract_elements(const Matrix matrix_a, const Matrix matrix_b) {
    size_t rows = matrix_a.rows;
    size_t cols = matrix_a.cols;

    Matrix result;
    
    if (matrix_a.rows != matrix_b.rows || matrix_a.cols != matrix_b.cols) {
        result = omp_matrix_nulled(rows, cols);

        return result;
    }

    result = matrix_init(rows, cols);

    size_t row, col;

    if (result.values != NULL) {
        #pragma omp parallel for private(row, col) shared(result, matrix_a, matrix_b)
        for(row = 0; row < rows; row++) {
            for(col = 0; col < cols; col++) {
                result.values[row][col] = matrix_a.values[row][col] - matrix_b.values[row][col];
            }
        }
    }

    return result;
}

Matrix omp_matrix_multiply_elements(const Matrix matrix_a, const Matrix matrix_b) {
    size_t rows = matrix_a.rows;
    size_t cols = matrix_a.cols;

    Matrix result;

    if (matrix_a.rows != matrix_b.rows || matrix_a.cols != matrix_b.cols) {
        result = omp_matrix_nulled(rows, cols);

        return result;
    }

    result = matrix_init(rows, cols);

    size_t row, col;

    if (result.values != NULL) {
        #pragma omp parallel for private(row, col) shared(result, matrix_a, matrix_b)
        for(row = 0; row < rows; row++) {
            for(col = 0; col < cols; col++) {
                result.values[row][col] = matrix_a.values[row][col] * matrix_b.values[row][col];
            }
        }
    }

    return result;
}