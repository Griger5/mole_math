#include <omp.h>

#include "../../../include/mole_math/matrix_properties.h"

#include "../../../include/mole_math/seq_matrix_properties.h"
#include "../../../include/mole_math/omp_matrix_properties.h"

double matrix_determinant(Matrix matrix, char flag) {
    size_t problem_size = matrix.rows * matrix.cols;

    double result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_determinant(matrix);
            break;
        case 's':
            result = seq_matrix_determinant(matrix);
            break;
        default:
            if ((double)problem_size/omp_get_num_procs() >= 6400/16.0) result = omp_matrix_determinant(matrix);
            else result = seq_matrix_determinant(matrix);
            break;
    }

    return result;
}