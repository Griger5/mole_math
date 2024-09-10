#ifndef OMP_MATRIX_TRANSFORM_H
#define OMP_MATRIX_TRANSFORM_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

void omp_matrix_switch_rows(Matrix *matrix, size_t row_1, size_t row_2);

void omp_matrix_subtract_rows(Matrix *matrix, size_t row_minuend, size_t row_subtrahend, double multiplier);

Matrix omp_matrix_inverse(const Matrix matrix);

#endif