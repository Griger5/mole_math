#ifndef SEQ_MATRIX_UTILS_H
#define SEQ_MATRIX_UTILS_H

#include "../main/matrix/matrix_define.h"

void matrix_print(const Matrix matrix);

Matrix matrix_identity(size_t N);

Matrix matrix_nulled(size_t rows, size_t cols);

Matrix matrix_copy(const Matrix matrix_to_copy);

Matrix matrix_array_to_matrix(double *array, size_t length);

#endif