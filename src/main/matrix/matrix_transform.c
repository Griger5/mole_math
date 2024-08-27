#include "../../../include/mole_math/matrix_transform.h"

#include "../../../include/mole_math/seq_matrix_transform.h"

void matrix_switch_rows(Matrix *matrix, size_t row_1, size_t row_2) {
    seq_matrix_switch_rows(matrix, row_1, row_2);
}

void matrix_subtract_rows(Matrix *matrix, size_t row_minuend, size_t row_subtrahend, double multiplier) {
    seq_matrix_subtract_rows(matrix, row_minuend, row_subtrahend, multiplier);
}

Matrix matrix_inverse(const Matrix matrix) {
    return seq_matrix_inverse(matrix);
}
