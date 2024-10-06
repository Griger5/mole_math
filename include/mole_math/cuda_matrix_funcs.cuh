#ifndef CUDA_MATRIX_FUNCS_H
#define CUDA_MATRIX_FUNCS_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

#ifdef __cplusplus
extern "C" {
#endif

double cuda_matrix_sum_row(const Matrix matrix, size_t row);

#ifdef __cplusplus
}
#endif

#endif