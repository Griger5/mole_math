#ifndef SEQ_MATRIX_TRANSFORM_H
#define SEQ_MATRIX_TRANSFORM_H

#include <mole_math/matrix_define.h>

void seq_matrix_switch_rows(Matrix *matrix, size_t row_1, size_t row_2);

void seq_matrix_subtract_rows(Matrix *matrix, size_t row_minuend, size_t row_subtrahend, double multiplier);

Matrix seq_matrix_inverse(const Matrix matrix);

#endif