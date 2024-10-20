#ifndef CUDA_MATRIX_PROPERTIES_H
#define CUDA_MATRIX_PROPERTIES_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

#ifdef __cplusplus
extern "C" {
#endif

double cuda_matrix_determinant(Matrix matrix);

#ifdef __cplusplus
}
#endif

#endif