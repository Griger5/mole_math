#ifndef SEQ_STATS_DATA_PROPERTIES_H
#define SEQ_STATS_DATA_PROPERTIES_H

#define PRIVATE_DATASET
#include <mole_math/stats_dataset_define.h>

double seq_stats_sum_data_field(DataSet dataset, size_t field_num);

double seq_stats_mean_arith_data_field(DataSet dataset, size_t field_num);

Matrix seq_stats_mean_diff_data_field(DataSet dataset, size_t field_num);

double seq_stats_std_var_biased_1d(DataSet dataset, size_t field_num);

double seq_stats_std_covar_popul(DataSet dataset, size_t field_1, size_t field_2);

#endif