#ifndef OMP_MATRIX_OPERATIONS_H
#define OMP_MATRIX_OPERATIONS_H

#include <mole_math/matrix_define.h>

Matrix omp_matrix_multiply(const Matrix matrix_a, const Matrix matrix_b);

Matrix omp_matrix_subtract_elements(const Matrix matrix_a, const Matrix matrix_b);

Matrix omp_matrix_multiply_elements(const Matrix matrix_a, const Matrix matrix_b);

#endif