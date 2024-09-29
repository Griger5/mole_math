#ifndef CUDA_MATRIX_TRANSFORM_H
#define CUDA_MATRIX_TRANSFORM_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_matrix_subtract_rows(Matrix *matrix, size_t row_minuend, size_t row_subtrahend, double multiplier);

#ifdef __cplusplus
}
#endif

#endif