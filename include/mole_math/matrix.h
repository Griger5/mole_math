#ifndef MOLEMATH_MATRIX_H
#define MOLEMATH_MATRIX_H

#include <mole_math/matrix_define.h>
#include <mole_math/macros.h>

#include <mole_math/matrix_funcs.h>

#define matrix_sum_row(MATRIX__matrix, SIZE_T__row_num, ...) matrix_sum_row(MATRIX__matrix, SIZE_T__row_num, GET_FLAG(__VA_ARGS__))

#include <mole_math/matrix_operations.h>

#define matrix_multiply(MATRIX__matrix_a, MATRIX__matrix_b, ...) matrix_multiply(MATRIX__matrix_a, MATRIX__matrix_b, GET_FLAG(__VA_ARGS__))
#define matrix_subtract_elements(MATRIX__matrix_a, MATRIX__matrix_b, ...) matrix_subtract_elements(MATRIX__matrix_a, MATRIX__matrix_b, GET_FLAG(__VA_ARGS__))
#define matrix_multiply_elements(MATRIX__matrix_a, MATRIX__matrix_b, ...) matrix_multiply_elements(MATRIX__matrix_a, MATRIX__matrix_b, GET_FLAG(__VA_ARGS__))

#include <mole_math/matrix_properties.h>

#define matrix_determinant(MATRIX__matrix, ...) matrix_determinant(MATRIX__matrix, GET_FLAG(__VA_ARGS__))
#define matrix_ij_minor(MATRIX__matrix, SIZE_T__i_row, SIZE_T__j_col, ...) matrix_ij_minor(MATRIX__matrix, SIZE_T__i_row, SIZE_T__j_col, GET_FLAG(__VA_ARGS__))
#define matrix_cofactor(MATRIX__matrix, ...) matrix_cofactor(MATRIX__matrix, GET_FLAG(__VA_ARGS__))

#include <mole_math/matrix_scalars.h>

#define matrix_subtract_scalar(MATRIX_PTR__matrix, DOUBLE__scalar, ...) matrix_subtract_scalar(MATRIX_PTR__matrix, DOUBLE__scalar, GET_FLAG(__VA_ARGS__))
#define matrix_multiply_row_scalar(MATRIX_PTR__matrix, SIZE_T__row_num, DOUBLE__scalar, ...) matrix_multiply_row_scalar(MATRIX_PTR__matrix, SIZE_T__row_num, DOUBLE__scalar, GET_FLAG(__VA_ARGS__))

#include <mole_math/matrix_transform.h>

#define matrix_switch_rows(MATRIX_PTR__matrix, SIZE_T__row_1, SIZE_T__row_2, ...) matrix_switch_rows(MATRIX_PTR__matrix, SIZE_T__row_1, SIZE_T__row_2, GET_FLAG(__VA_ARGS__))
#define matrix_subtract_rows(MATRIX_PTR__matrix, SIZE_T__row_minuend, SIZE_T__row_subtrahend, DOUBLE__multiplier, ...) matrix_subtract_rows(MATRIX_PTR__matrix, SIZE_T__row_minuend, SIZE_T__row_subtrahend, DOUBLE__multiplier, GET_FLAG(__VA_ARGS__))
#define matrix_transpose(MATRIX__matrix, ...) matrix_transpose(MATRIX__matrix, GET_FLAG(__VA_ARGS__))
#define matrix_ij_minor_matrix(MATRIX__matrix, SIZE_T__i_row, SIZE_T__j_col, ...) matrix_ij_minor_matrix(MATRIX__matrix, SIZE_T__i_row, SIZE_T__j_col, GET_FLAG(__VA_ARGS__))
#define matrix_inverse(MATRIX__matrix, ...) matrix_inverse(MATRIX__matrix, GET_FLAG(__VA_ARGS__))

#include <mole_math/matrix_utils.h>

#define matrix_identity(SIZE_T__n, ...) matrix_identity(SIZE_T__n, GET_FLAG(__VA_ARGS__))
#define matrix_random(SIZE_T__rows, SIZE_T__cols, ...) matrix_random(SIZE_T__rows, SIZE_T__cols, GET_FLAG(__VA_ARGS__))
#define matrix_init_integers(SIZE_T__rows, SIZE_T__cols, ...) matrix_init_integers(SIZE_T__rows, SIZE_T__cols, GET_FLAG(__VA_ARGS__))
#define matrix_copy(MATRIX__matrix_to_copy, ...) matrix_copy(MATRIX__matrix_to_copy, GET_FLAG(__VA_ARGS__))
#define matrix_replace(MATRIX_PTR__to_replace, MATRIX__matrix_to_copy, ...) matrix_replace(MATRIX_PTR__to_replace, MATRIX__matrix_to_copy, GET_FLAG(__VA_ARGS__))
#define matrix_array_to_matrix(DOUBLE_PTR__array, SIZE_T__length, ...) matrix_array_to_matrix(DOUBLE_PTR__array, SIZE_T__length, GET_FLAG(__VA_ARGS__)) 

#endif