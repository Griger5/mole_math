#ifndef SEQ_MATRIX_UTILS_H
#define SEQ_MATRIX_UTILS_H

#include <mole_math/matrix_define.h>

void matrix_print(const Matrix matrix);

Matrix seq_matrix_identity(size_t N);

Matrix seq_matrix_nulled(size_t rows, size_t cols);

Matrix seq_matrix_copy(const Matrix matrix_to_copy);

Matrix seq_matrix_array_to_matrix(double *array, size_t length);

#endif