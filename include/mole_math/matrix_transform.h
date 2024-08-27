#ifndef MATRIX_TRANSFORM_H
#define MATRIX_TRANSFORM_H

#include <mole_math/matrix_define.h>

void matrix_switch_rows(Matrix *matrix, size_t row_1, size_t row_2);

void matrix_subtract_rows(Matrix *matrix, size_t row_minuend, size_t row_subtrahend, double multiplier);

Matrix matrix_inverse(const Matrix matrix);

#endif