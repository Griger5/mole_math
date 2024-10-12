#ifndef OMP_MATRIX_TRANSFORM_H
#define OMP_MATRIX_TRANSFORM_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

void omp_matrix_subtract_rows(Matrix *matrix, size_t row_minuend, size_t row_subtrahend, double multiplier);

Matrix omp_matrix_transpose(const Matrix matrix);

Matrix omp_matrix_ij_minor_matrix(const Matrix matrix, size_t i_row, size_t j_col);

Matrix omp_matrix_inverse(Matrix matrix);

#endif