#include <math.h>
#include <omp.h>

#include "../../include/mole_math/omp_matrix_funcs.h"

double omp_matrix_sum_row(const Matrix matrix, size_t row) {
    size_t cols = matrix.cols;

    double sum;

    if (row >= matrix.rows) return NAN;

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < cols; i++) { 
        {
            sum += matrix.values[row][i];
        }
    }

    return sum;
}