#ifndef MATRIX_DEFINE_H
#define MATRIX_DEFINE_H

#include <stddef.h>

typedef struct _Matrix {
    size_t rows;
    size_t cols;
    double **values;
} Matrix;

Matrix matrix_init(size_t rows, size_t cols);

void matrix_free(Matrix *matrix);

#define MFREE(MATRIX__matrix) matrix_free(&MATRIX__matrix)

#endif