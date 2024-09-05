#include <math.h>

#include "../../include/mole_math/seq_matrix_funcs.h"

double seq_matrix_sum_row(const Matrix matrix, size_t row) {
    size_t cols = matrix.cols;

    double sum = 0;

    if (row >= matrix.rows) return NAN;

    for (size_t i = 0; i < cols; i++) {
        sum += matrix.values[row][i];
    }

    return sum;
}