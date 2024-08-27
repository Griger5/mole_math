#include <math.h>

#include "seq_matrix_properties.h"
#include "seq_matrix_transform.h"
#include "seq_matrix_utils.h"

double seq_matrix_determinant(Matrix matrix) {   
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

    Matrix matrix_copied = seq_matrix_copy(matrix);

    double ratio;
    double determinant = 1;

    if (matrix_copied.values[0][0] == 0) {
        for (size_t i = 1; i < N; i++) {
            if (matrix_copied.values[i][i] != 0) {
                seq_matrix_switch_rows(&matrix_copied, 0, i);
                determinant *= -1;
                break;
            }       
        }
    }

    for (size_t i = 0; i < N; i++) {
        if (matrix_copied.values[i][i] == 0) return 0;

        for (size_t j = i+1; j < N; j++) {
            ratio = matrix_copied.values[j][i] / matrix_copied.values[i][i];
            seq_matrix_subtract_rows(&matrix_copied, j, i, ratio);
        }
        
        determinant *= matrix_copied.values[i][i];
    }

    matrix_free(&matrix_copied);

    return determinant;
}