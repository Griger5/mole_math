#ifndef SEQ_MATRIX_OPERATIONS_H
#define SEQ_MATRIX_OPERATIONS_H

#include "../main/matrix/matrix_define.h"

Matrix matrix_multiply(const Matrix matrix_a, const Matrix matrix_b);

Matrix matrix_subtract_elements(const Matrix matrix_a, const Matrix matrix_b);

Matrix matrix_multiply_elements(const Matrix matrix_a, const Matrix matrix_b);

#endif