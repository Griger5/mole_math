#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../../include/mole_math/seq_matrix_utils.h"

Matrix seq_matrix_identity(size_t N) {
    Matrix identity = matrix_init(N, N);

    if (identity.values != NULL) {
        for (size_t i = 0; i < N; i++) {
            identity.values[i][i] = 1.0;
        }
    }

    return identity;
}

Matrix seq_matrix_nulled(size_t rows, size_t cols) {
    Matrix matrix;
    
    matrix.rows = rows;
    matrix.cols = cols;

    matrix.values = NULL;
    matrix.determinant = NULL;

    return matrix;
}

Matrix seq_matrix_random(size_t rows, size_t cols) {
    Matrix random_mat = matrix_init(rows, cols);

    if (random_mat.values != NULL) {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                random_mat.values[i][j] = (double)rand()/RAND_MAX;
            }
        }
    }

    return random_mat;
}

Matrix seq_matrix_init_integers(size_t rows, size_t cols) {
    Matrix int_mat = matrix_init(rows, cols);

    if (int_mat.values != NULL) {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                int_mat.values[i][j] = i*cols + j + 1;
            }
        }
    }

    return int_mat;
}

Matrix seq_matrix_copy(const Matrix matrix_to_copy) {
    size_t rows = matrix_to_copy.rows;
    size_t cols = matrix_to_copy.cols;

    Matrix copy;
    if (matrix_to_copy.values == NULL) copy = seq_matrix_nulled(rows, cols);
    else copy = matrix_init(rows, cols);
    
    if (copy.values != NULL) {
        /* if (matrix_to_copy.values == NULL) copy.values = NULL;
        else {
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    copy.values[i][j] = matrix_to_copy.values[i][j];
                }
            }
        } */
        size_t matrix_size_bytes = rows * cols * sizeof(double);

        memcpy(copy.values[0], matrix_to_copy.values[0], matrix_size_bytes);
    }

    if (matrix_to_copy.determinant != NULL) {
        if (!isinf(*matrix_to_copy.determinant)) *copy.determinant = *matrix_to_copy.determinant;
    }

    return copy;
}

void seq_matrix_replace(Matrix *to_replace, const Matrix to_copy) {
    matrix_free(to_replace);

    *to_replace = seq_matrix_copy(to_copy);
}

Matrix seq_matrix_array_to_matrix(double *array, size_t length) {
    Matrix matrix = matrix_init(1, length);

    if (matrix.values != NULL) {
        for (size_t i = 0; i < length; i++) {
            matrix.values[0][i] = array[i];
        }
    }
    
    return matrix;
}