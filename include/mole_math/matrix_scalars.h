#ifndef MATRIX_SCALARS_H
#define MATRIX_SCALARS_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

void matrix_subtract_scalar(Matrix *matrix, double scalar, char flag);

void matrix_multiply_row_scalar(Matrix *matrix, size_t row_num, double scalar, char flag);

#endif