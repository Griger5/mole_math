#ifndef SEQ_MATRIX_SCALARS_H
#define SEQ_MATRIX_SCALARS_H

#include <mole_math/matrix_define.h>

void seq_matrix_subtract_scalar(Matrix *matrix, double scalar);

void seq_matrix_multiply_row_scalar(Matrix *matrix, size_t row_num, double scalar);

#endif