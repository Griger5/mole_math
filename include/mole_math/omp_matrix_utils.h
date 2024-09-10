#ifndef OMP_MATRIX_UTILS_H
#define OMP_MATRIX_UTILS_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

Matrix omp_matrix_identity(size_t N);

Matrix omp_matrix_nulled(size_t rows, size_t cols);

Matrix omp_matrix_random(size_t rows, size_t cols);

Matrix omp_matrix_copy(const Matrix matrix_to_copy);

void omp_matrix_replace(Matrix *to_replace, const Matrix matrix_to_copy);

Matrix omp_matrix_array_to_matrix(double *array, size_t length);

#endif