#include "../../../include/mole_math/matrix_operations.h"

#include "../../../include/mole_math/seq_matrix_operations.h"


Matrix matrix_multiply(const Matrix matrix_a, const Matrix matrix_b) {
    return seq_matrix_multiply(matrix_a, matrix_b);
}

Matrix matrix_subtract_elements(const Matrix matrix_a, const Matrix matrix_b) {
    return seq_matrix_subtract_elements(matrix_a, matrix_b);
}

Matrix matrix_multiply_elements(const Matrix matrix_a, const Matrix matrix_b) {
    return seq_matrix_multiply_elements(matrix_a, matrix_b);
}