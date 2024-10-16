#ifndef OMP_MATRIX_UTILS_H
#define OMP_MATRIX_UTILS_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

Matrix omp_matrix_identity(size_t N);

Matrix omp_matrix_random(size_t rows, size_t cols);

Matrix omp_matrix_init_integers(size_t rows, size_t cols);

#endif