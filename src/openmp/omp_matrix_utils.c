#include <stdlib.h>
#include <math.h>

#include "../../include/mole_math/omp_matrix_utils.h"

Matrix omp_matrix_identity(size_t N) {
    Matrix identity = matrix_init(N, N);

    if (identity.values != NULL) {
        #pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
            identity.values[i][i] = 1.0;
        }
    }           

    return identity;
}

Matrix omp_matrix_nulled(size_t rows, size_t cols) {
    Matrix matrix;
    
    matrix.rows = rows;
    matrix.cols = cols;

    matrix.values = NULL;
    matrix.determinant = NULL;

    return matrix;
}

Matrix omp_matrix_random(size_t rows, size_t cols) {
    Matrix random_mat = matrix_init(rows, cols);

    size_t j;

    if (random_mat.values != NULL) {
        #pragma omp parallel for private(j)
        for (size_t i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                random_mat.values[i][j] = (double)rand()/RAND_MAX;
            }
        }
    }

    return random_mat;
}

Matrix omp_matrix_init_integers(size_t rows, size_t cols) {
    Matrix int_mat = matrix_init(rows, cols);

    size_t j;

    if (int_mat.values != NULL) {
        #pragma omp parallel for private(j)
        for (size_t i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                int_mat.values[i][j] = i*cols + j + 1;
            }
        }
    }

    return int_mat;
}

Matrix omp_matrix_copy(const Matrix matrix_to_copy) {
    size_t rows = matrix_to_copy.rows;
    size_t cols = matrix_to_copy.cols;

    Matrix copy;
    if (matrix_to_copy.values == NULL) copy = omp_matrix_nulled(rows, cols);
    else copy = matrix_init(rows, cols);
    
    if (copy.values != NULL) {
        if (matrix_to_copy.values == NULL) copy.values = NULL;
        else {
            size_t j;
            #pragma omp parallel for private(j)
            for (size_t i = 0; i < rows; i++) {
                for (j = 0; j < cols; j++) {
                    copy.values[i][j] = matrix_to_copy.values[i][j];
                }
            }
        }
    }

    return copy;
}

void omp_matrix_replace(Matrix *to_replace, const Matrix to_copy) {
    matrix_free(to_replace);

    *to_replace = omp_matrix_copy(to_copy);
}

Matrix omp_matrix_array_to_matrix(double *array, size_t length) {
    Matrix matrix = matrix_init(1, length);

    if (matrix.values != NULL) {
        #pragma omp parallel for
        for (size_t i = 0; i < length; i++) {
            matrix.values[0][i] = array[i];
        }
    }
    
    return matrix;
}