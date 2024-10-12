#ifndef CUDA_MATRIX_SCALARS_H
#define CUDA_MATRIX_SCALARS_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_matrix_subtract_scalar(Matrix *matrix, double scalar);

void cuda_matrix_multiply_row_scalar(Matrix *matrix, size_t row_num, double scalar);

#ifdef __cplusplus
}
#endif

#endif