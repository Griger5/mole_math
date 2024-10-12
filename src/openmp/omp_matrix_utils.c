#include <stdlib.h>
#include <math.h>

#include "../../include/mole_math/omp_matrix_utils.h"
#include "../../include/mole_math/seq_matrix_utils.h"

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

void omp_matrix_replace(Matrix *to_replace, const Matrix to_copy) {
    matrix_free(to_replace);

    *to_replace = seq_matrix_copy(to_copy);
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