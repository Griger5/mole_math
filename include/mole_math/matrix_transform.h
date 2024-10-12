#ifndef MATRIX_TRANSFORM_H
#define MATRIX_TRANSFORM_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

void matrix_switch_rows(Matrix *matrix, size_t row_1, size_t row_2);

void matrix_subtract_rows(Matrix *matrix, size_t row_minuend, size_t row_subtrahend, double multiplier, char flag);

Matrix matrix_transpose(const Matrix matrix, char flag);

Matrix matrix_ij_minor_matrix(const Matrix matrix, size_t i_row, size_t j_col, char flag);

Matrix matrix_inverse(const Matrix matrix, char flag);

#endif