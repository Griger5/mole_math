#ifndef MATRIX_DEFINE_H
#define MATRIX_DEFINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _Matrix {
    double **values;
    struct {
        size_t rows;
        size_t cols;
        double *determinant; // pointer, so it can be modified with pass by value
    } PRIVATE_MAT;
} Matrix;

Matrix matrix_init(size_t rows, size_t cols);

void matrix_free(Matrix *matrix);

size_t matrix_get_rows(const Matrix matrix);

size_t matrix_get_cols(const Matrix matrix);

double matrix_get_determinant(const Matrix matrix);

void matrix_reset_properties(Matrix *matrix);

#define MFREE(MATRIX__matrix) matrix_free(&MATRIX__matrix)

#ifdef __cplusplus
}
#endif

#endif