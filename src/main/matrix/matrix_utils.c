#include "../../../include/mole_math/matrix_utils.h"

#include "../../../include/mole_math/seq_matrix_utils.h"

void matrix_print(const Matrix matrix) {
    seq_matrix_print(matrix);
}

Matrix matrix_identity(size_t N) {
    return seq_matrix_identity(N);
}

Matrix matrix_nulled(size_t rows, size_t cols) {
    return seq_matrix_nulled(rows, cols);
}

Matrix matrix_copy(const Matrix matrix_to_copy) {
    return seq_matrix_copy(matrix_to_copy);
}

Matrix matrix_array_to_matrix(double *array, size_t length) {
    return seq_matrix_array_to_matrix(array, length);
}