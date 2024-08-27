#include "../../../include/mole_math/matrix_scalars.h"

#include "../../../include/mole_math/seq_matrix_scalars.h"

void matrix_subtract_scalar(Matrix *matrix, double scalar) {
    seq_matrix_subtract_scalar(matrix, scalar);
}

void matrix_multiply_row_scalar(Matrix *matrix, size_t row_num, double scalar) {
    seq_matrix_multiply_row_scalar(matrix, row_num, scalar);
}