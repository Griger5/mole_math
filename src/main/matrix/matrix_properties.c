#include "../../../include/mole_math/matrix_properties.h"

#include "../../../include/mole_math/seq_matrix_properties.h"

double matrix_determinant(Matrix matrix) {
    return seq_matrix_determinant(matrix);
}