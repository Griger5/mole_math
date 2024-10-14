#ifndef CUDA_MATRIX_UTILS_H
#define CUDA_MATRIX_UTILS_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

#ifdef __cplusplus
extern "C" {
#endif

Matrix cuda_matrix_identity(size_t N);

Matrix cuda_matrix_random(size_t rows, size_t cols);

Matrix cuda_matrix_init_integers(size_t rows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif