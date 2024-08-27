#ifndef SEQ_MATRIX_SCALARS_H
#define SEQ_MATRIX_SCALARS_H

#include "../main/matrix/matrix_define.h"

void matrix_subtract_scalar(Matrix *matrix, double scalar);

void matrix_multiply_row_scalar(Matrix *matrix, size_t row_num, double scalar);

#endif