#ifndef CUDA_MATRIX_OPERATIONS_H
#define CUDA_MATRIX_OPERATIONS_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

#ifdef __cplusplus
extern "C" {
#endif

Matrix cuda_matrix_multiply(const Matrix matrix_a, const Matrix matrix_b);

#ifdef __cplusplus
}
#endif

#endif