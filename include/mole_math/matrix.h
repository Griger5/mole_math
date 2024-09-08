#include <mole_math/matrix_define.h>
#include <mole_math/macros.h>

#include <mole_math/matrix_funcs.h>

#define matrix_sum_row(MATRIX__matrix, SIZE_T__row_num, ...) matrix_sum_row(MATRIX__matrix, SIZE_T__row_num, GET_FLAG(__VA_ARGS__))

#include <mole_math/matrix_operations.h>
#include <mole_math/matrix_properties.h>
#include <mole_math/matrix_scalars.h>
#include <mole_math/matrix_transform.h>
#include <mole_math/matrix_utils.h>