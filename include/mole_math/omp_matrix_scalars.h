#ifndef OMP_MATRIX_SCALARS_H
#define OMP_MATRIX_SCALARS_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

void omp_matrix_subtract_scalar(Matrix *matrix, double scalar);

void omp_matrix_multiply_row_scalar(Matrix *matrix, size_t row_num, double scalar);

#endif