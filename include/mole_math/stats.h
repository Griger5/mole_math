#ifndef MOLEMATH_STATS_H
#define MOLEMATH_STATS_H

#include <mole_math/stats_dataset_define.h>
#include <mole_math/macros.h>

#include <mole_math/stats_data_properties.h>

#define stats_sum_data_field(DATASET__dataset, SIZE_T__field_num, ...) stats_sum_data_field(DATASET__dataset, SIZE_T__field_num, GET_FLAG(__VA_ARGS__))
#define stats_mean_arith_data_field(DATASET__dataset, SIZE_T__field_num, ...) stats_mean_arith_data_field(DATASET__dataset, SIZE_T__field_num, GET_FLAG(__VA_ARGS__))
#define stats_mean_diff_data_field(DATASET__dataset, SIZE_T__field_num, ...) stats_mean_diff_data_field(DATASET__dataset, SIZE_T__field_num, GET_FLAG(__VA_ARGS__))

#undef PRIVATE_DATASET
#define GET_DATA PRIVATE_DATASET.data.values

#endif