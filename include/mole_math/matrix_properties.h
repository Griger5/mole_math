#ifndef MATRIX_PROPERTIES_H
#define MATRIX_PROPERTIES_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

double matrix_determinant(Matrix matrix, char flag);

double matrix_ij_minor(const Matrix matrix, size_t i_row, size_t j_col, char flag);

Matrix matrix_cofactor(const Matrix matrix, char flag);

#endif