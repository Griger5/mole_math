#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <mole_math/matrix_define.h>

Matrix matrix_multiply(const Matrix matrix_a, const Matrix matrix_b, char flag);

Matrix matrix_subtract_elements(const Matrix matrix_a, const Matrix matrix_b, char flag);

Matrix matrix_multiply_elements(const Matrix matrix_a, const Matrix matrix_b, char flag);

#endif