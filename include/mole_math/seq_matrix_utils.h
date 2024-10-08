#ifndef SEQ_MATRIX_UTILS_H
#define SEQ_MATRIX_UTILS_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

#ifdef __cplusplus
extern "C" {
#endif

Matrix seq_matrix_identity(size_t N);

Matrix seq_matrix_nulled(size_t rows, size_t cols);

Matrix seq_matrix_random(size_t rows, size_t cols);

Matrix seq_matrix_init_integers(size_t rows, size_t cols);

Matrix seq_matrix_copy(const Matrix matrix_to_copy);

void seq_matrix_replace(Matrix *to_replace, const Matrix matrix_to_copy);

Matrix seq_matrix_array_to_matrix(double *array, size_t length);

#ifdef __cplusplus
}
#endif

#endif