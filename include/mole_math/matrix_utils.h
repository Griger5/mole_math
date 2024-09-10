#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

void matrix_print(const Matrix matrix);

Matrix matrix_identity(size_t N, char flag);

Matrix matrix_nulled(size_t rows, size_t cols);

Matrix matrix_random(size_t rows, size_t cols, char flag);

Matrix matrix_copy(const Matrix matrix_to_copy, char flag);

void matrix_replace(Matrix *to_replace, const Matrix matrix_to_copy, char flag);

Matrix matrix_array_to_matrix(double *array, size_t length, char flag);

#endif