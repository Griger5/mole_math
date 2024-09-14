#ifndef SEQ_MATRIX_PROPERTIES_H
#define SEQ_MATRIX_PROPERTIES_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

double seq_matrix_determinant(Matrix matrix);

double seq_matrix_ij_minor(const Matrix matrix, size_t i_row, size_t j_col);

Matrix seq_matrix_cofactor(const Matrix matrix);

#endif