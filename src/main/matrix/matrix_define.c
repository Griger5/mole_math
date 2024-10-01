#include <stdlib.h>
#include <math.h>

#define PRIVATE_MAT
#include "../../../include/mole_math/matrix_define.h"

Matrix matrix_init(size_t rows, size_t cols) {
    Matrix matrix;

    matrix.rows = rows;
    matrix.cols = cols;

    matrix.values = malloc(rows * sizeof(double *));

    if (matrix.values != NULL) {
        matrix.values[0] = calloc(rows, cols * sizeof(double));

        if (matrix.values[0] != NULL) {
            for (size_t i = 1; i < rows; i++) {
                matrix.values[i] = matrix.values[0] + i*cols;
            }
        }
        else matrix.values = NULL;
    }

    matrix.determinant = malloc(2*sizeof(double));
    
    if (matrix.determinant != NULL) *matrix.determinant = INFINITY; // sentinel value

    return matrix;
}

void matrix_free(Matrix *matrix) {
    if (matrix->values != NULL) {
        free(matrix->values[0]);
        free(matrix->values);
    }

    if (matrix->determinant != NULL) {
        free(matrix->determinant);
        matrix->determinant = NULL;
    }
    //matrix->determinant = NULL;

    matrix->values = NULL;
}

size_t matrix_get_rows(const Matrix matrix) {
    return matrix.rows;
}

size_t matrix_get_cols(const Matrix matrix) {
    return matrix.cols;
}

double matrix_get_determinant(const Matrix matrix) {
    return *matrix.determinant;
}

void matrix_reset_properties(Matrix *matrix) {
    *matrix->determinant = INFINITY;
}