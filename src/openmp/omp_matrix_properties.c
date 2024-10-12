#include <math.h>
#include <omp.h>

#include "../../include/mole_math/omp_matrix_properties.h"
#include "../../include/mole_math/omp_matrix_transform.h"
#include "../../include/mole_math/seq_matrix_transform.h"
#include "../../include/mole_math/seq_matrix_utils.h"

double omp_matrix_determinant(Matrix matrix) {   
    if (matrix.rows != matrix.cols) return NAN;

    size_t N = matrix.rows;
    double determinant = 1;
    
    if (N == 1) {
        determinant = matrix.values[0][0];
        *matrix.determinant = determinant;

        return determinant;
    }

    if (N == 2) {
        determinant = (matrix.values[0][0] * matrix.values[1][1] - matrix.values[0][1] * matrix.values[1][0]);
        *matrix.determinant = determinant;

        return determinant;    
    }

    if (N == 3) {

        double aei = matrix.values[0][0] * matrix.values[1][1] * matrix.values[2][2];
        double bfg = matrix.values[0][1] * matrix.values[1][2] * matrix.values[2][0];
        double cdh = matrix.values[0][2] * matrix.values[1][0] * matrix.values[2][1];

        double ceg = matrix.values[0][2] * matrix.values[1][1] * matrix.values[2][0];
        double bdi = matrix.values[0][1] * matrix.values[1][0] * matrix.values[2][2];
        double afh = matrix.values[0][0] * matrix.values[1][2] * matrix.values[2][1];

        determinant = aei + bfg + cdh - ceg - bdi - afh;
        *matrix.determinant = determinant;

        return determinant;
    }

    Matrix matrix_copied = seq_matrix_copy(matrix);

    double ratio;
    double sign = 1;

    if (matrix_copied.values[0][0] == 0) {
        for (size_t i = 1; i < N; i++) {
            if (matrix_copied.values[i][i] != 0) {
                seq_matrix_switch_rows(&matrix_copied, 0, i);
                sign *= -1;
                break;
            }       
        }
    }

    size_t i;
    int det_is_zero = 0;

    #pragma omp parallel private(i,ratio)
    {
        for (i = 0; i < N; i++) {
            if (det_is_zero) continue;

            if (matrix_copied.values[i][i] == 0) {
                det_is_zero = 1;
                determinant = 0;
            }
            
            #pragma omp for
            for (size_t j = i+1; j < N; j++) {
                ratio = matrix_copied.values[j][i] / matrix_copied.values[i][i];
                omp_matrix_subtract_rows(&matrix_copied, j, i, ratio);
            }

            #pragma omp single
            determinant *= matrix_copied.values[i][i];
        }
    }

    matrix_free(&matrix_copied);

    *matrix.determinant = determinant;

    return sign*determinant;
}

double omp_matrix_ij_minor(const Matrix matrix, size_t i_row, size_t j_col) {
    Matrix minor_matrix = omp_matrix_ij_minor_matrix(matrix, i_row, j_col);

    if (minor_matrix.values == NULL) return NAN;

    double minor = omp_matrix_determinant(minor_matrix);

    matrix_free(&minor_matrix);

    return minor;
}

Matrix omp_matrix_cofactor(const Matrix matrix) {
    size_t rows = matrix.rows;
    size_t cols = matrix.cols;

    if (matrix.values == NULL) return seq_matrix_nulled(rows, cols);

    Matrix cofactor_matrix = matrix_init(rows, cols);
    int sign;
    size_t j;

    if (cofactor_matrix.values != NULL) {
        #pragma omp parallel for private(j)
        for (size_t i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                sign = 1 - 2 * ((i+j)%2);
                cofactor_matrix.values[i][j] = omp_matrix_ij_minor(matrix, i, j) * sign;
            }
        }
    }

    return cofactor_matrix;
}