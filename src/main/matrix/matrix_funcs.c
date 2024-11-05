#include <omp.h>

#include "../../../include/mole_math/matrix_funcs.h"

#include "../../../include/mole_math/seq_matrix_funcs.h"
#include "../../../include/mole_math/omp_matrix_funcs.h"
#include "../../../include/mole_math/cuda_matrix_funcs.cuh"

double matrix_sum_row(Matrix matrix, size_t row_num, char flag) {
    size_t problem_size = matrix.cols;

    double sum;
    
    switch (flag) {
        case 'o':
            sum = omp_matrix_sum_row(matrix, row_num);
            break;
        case 's':
            sum = seq_matrix_sum_row(matrix, row_num);
            break;
        #ifdef CUDA_SM_COUNT
        case 'c':
            sum = cuda_matrix_sum_row(matrix, row_num);
            break;
        #endif
        default:
            if ((double)problem_size/omp_get_num_procs() >= 4000/16.0) sum = omp_matrix_sum_row(matrix, row_num);
            else sum = seq_matrix_sum_row(matrix, row_num);
            break;
    }

    return sum;
}