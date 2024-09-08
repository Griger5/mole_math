#include <math.h>
#include <omp.h>

#include "../../include/mole_math/omp_matrix_properties.h"
#include "../../include/mole_math/omp_matrix_transform.h"
#include "../../include/mole_math/omp_matrix_utils.h"

double omp_matrix_determinant(Matrix matrix) {   
    if (matrix.rows != matrix.cols) return NAN;

    size_t N = matrix.rows;
    
    if (N == 1) return matrix.values[0][0];

    if (N == 2) return (matrix.values[0][0] * matrix.values[1][1] - matrix.values[0][1] * matrix.values[1][0]);

    if (N == 3) {

        double aei = matrix.values[0][0] * matrix.values[1][1] * matrix.values[2][2];
        double bfg = matrix.values[0][1] * matrix.values[1][2] * matrix.values[2][0];
        double cdh = matrix.values[0][2] * matrix.values[1][0] * matrix.values[2][1];

        double ceg = matrix.values[0][2] * matrix.values[1][1] * matrix.values[2][0];
        double bdi = matrix.values[0][1] * matrix.values[1][0] * matrix.values[2][2];
        double afh = matrix.values[0][0] * matrix.values[1][2] * matrix.values[2][1];

        return aei + bfg + cdh - ceg - bdi - afh;
    }

    Matrix matrix_copied = omp_matrix_copy(matrix);

    double ratio;
    double sign = 1;
    double determinant = 1;

    if (matrix_copied.values[0][0] == 0) {
        for (size_t i = 1; i < N; i++) {
            if (matrix_copied.values[i][i] != 0) {
                omp_matrix_switch_rows(&matrix_copied, 0, i);
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

    return sign*determinant;
}