#ifndef SEQ_MATRIX_OPERATIONS_H
#define SEQ_MATRIX_OPERATIONS_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

Matrix seq_matrix_multiply(const Matrix matrix_a, const Matrix matrix_b);

Matrix seq_matrix_subtract_elements(const Matrix matrix_a, const Matrix matrix_b);

Matrix seq_matrix_multiply_elements(const Matrix matrix_a, const Matrix matrix_b);

#endif