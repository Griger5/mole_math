#include "../../../include/mole_math/matrix_funcs.h"

#include "../../../include/mole_math/seq_matrix_funcs.h"

double matrix_sum_row(Matrix matrix, size_t row) {
    return seq_matrix_sum_row(matrix, row);
}