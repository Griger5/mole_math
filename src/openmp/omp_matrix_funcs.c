#include <math.h>
#include <omp.h>

#include "../../include/mole_math/omp_matrix_funcs.h"

double omp_matrix_sum_row(const Matrix matrix, size_t row) {
    size_t cols = matrix.cols;

    double sum = 0;

    if (row >= matrix.rows) return NAN;

    size_t i;
    double thread_sum = 0;

    #pragma omp parallel private(i, thread_sum) shared(matrix) 
    {
        #pragma omp for
        for (i = 0; i < cols; i++) { 
            {
                thread_sum += matrix.values[row][i];
            }
        }

        #pragma omp critical 
        {
            sum += thread_sum;
        }
    }

    return sum;
}