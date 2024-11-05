#include <omp.h>
#include <math.h>

#include "../../../include/mole_math/matrix_properties.h"

#include "../../../include/mole_math/seq_matrix_properties.h"
#include "../../../include/mole_math/omp_matrix_properties.h"
#include "../../../include/mole_math/cuda_matrix_properties.cuh"

double matrix_determinant(Matrix matrix, char flag) {
    size_t problem_size = matrix.rows * matrix.cols;

    double result;
    
    if (!isinf(*matrix.determinant)) return *matrix.determinant;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_determinant(matrix);
            break;
        case 's':
            result = seq_matrix_determinant(matrix);
            break;
        #ifdef CUDA_SM_COUNT
        case 'c':
            result = cuda_matrix_determinant(matrix);
            break;
        #endif
        default:
            if ((double)problem_size/omp_get_num_procs() >= 6400/16.0) result = omp_matrix_determinant(matrix);
            else result = seq_matrix_determinant(matrix);
            break;
    }

    return result;
}

double matrix_ij_minor(const Matrix matrix, size_t i_row, size_t j_col, char flag) {
    size_t problem_size = matrix.rows * matrix.cols;

    double result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_ij_minor(matrix, i_row, j_col);
            break;
        case 's':
            result = seq_matrix_ij_minor(matrix, i_row, j_col);
            break;
        default:
            if ((double)problem_size/omp_get_num_procs() >= 4096/16.0) result = omp_matrix_ij_minor(matrix, i_row, j_col);
            else result = seq_matrix_ij_minor(matrix, i_row, j_col);
            break;
    }

    return result;
}

Matrix matrix_cofactor(const Matrix matrix, char flag) {
    size_t problem_size = matrix.rows * matrix.cols;

    Matrix result;
    
    switch (flag) {
        case 'o':
            result = omp_matrix_cofactor(matrix);
            break;
        case 's':
            result = seq_matrix_cofactor(matrix);
            break;
        default:
            if ((double)problem_size/omp_get_num_procs() >= 400/16.0) result = omp_matrix_cofactor(matrix);
            else result = seq_matrix_cofactor(matrix);
            break;
    }

    return result;
}