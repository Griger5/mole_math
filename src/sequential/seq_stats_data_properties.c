#include "../../include/mole_math/seq_stats_data_properties.h"

double seq_stats_sum_data_field(DataSet dataset, size_t field_num) {
    size_t rec_count = dataset.rec_count;
    double sum = 0;

    for (size_t i = 0; i < rec_count; i++) {
        sum += dataset.data.values[i][field_num]; 
    }

    return sum;
}

double seq_stats_mean_arith_data_field(DataSet dataset, size_t field_num) {
    double sum = seq_stats_sum_data_field(dataset, field_num);

    return sum/dataset.rec_count;
}

Matrix seq_stats_mean_diff_data_field(DataSet dataset, size_t field_num) {
    double mean_x = seq_stats_mean_arith_data_field(dataset, field_num);

    size_t rec_count = dataset.rec_count;

    Matrix mean_diff_vector = matrix_init(1, rec_count);

    for (size_t i = 0; i < rec_count; i++) {
        mean_diff_vector.values[0][i] = dataset.data.values[i][field_num] - mean_x;
    }

    return mean_diff_vector;
}