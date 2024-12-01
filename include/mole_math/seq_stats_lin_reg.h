#ifndef SEQ_STATS_LIN_REG_H
#define SEQ_STATS_LIN_REG_H

#define PRIVATE_DATASET
#include <mole_math/stats_dataset_define.h>

#include <mole_math/stats_lin_reg.h>

LinParam seq_stats_lin_reg_simple(DataSet dataset, size_t dim1, size_t dim2);

#endif