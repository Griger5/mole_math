#ifndef STATS_LIN_REG_H
#define STATS_LIN_REG_H

#define PRIVATE_DATASET
#include <mole_math/stats_dataset_define.h>

typedef struct _LinParam {
    double slope;
    double intercept;
} LinParam;

LinParam stats_lin_reg_simple(DataSet dataset, size_t dim1, size_t dim2, char flag);

#endif