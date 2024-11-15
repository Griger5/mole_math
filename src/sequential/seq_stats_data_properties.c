#include "../../include/mole_math/seq_stats_data_properties.h"

#include "../../include/mole_math/seq_matrix_funcs.h"
#include "../../include/mole_math/seq_matrix_operations.h"

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

double seq_stats_std_var_biased_1d(DataSet dataset, size_t field_num) {
    Matrix mean_diff_vector = seq_stats_mean_diff_data_field(dataset, field_num);
    Matrix mean_diff_vector_square = seq_matrix_multiply_elements(mean_diff_vector, mean_diff_vector);

    double sum_mean_diff = seq_matrix_sum_row(mean_diff_vector_square, 0);

    MFREE(mean_diff_vector);
    MFREE(mean_diff_vector_square);

    return sum_mean_diff/dataset.rec_count;
}

double seq_stats_std_covar_popul(DataSet dataset, size_t field_1, size_t field_2) {
    Matrix mean_diff_vector1 = seq_stats_mean_diff_data_field(dataset, field_1);
    Matrix mean_diff_vector2 = seq_stats_mean_diff_data_field(dataset, field_2);

    Matrix multiplied_vectors = seq_matrix_multiply_elements(mean_diff_vector1, mean_diff_vector2);
    double sum_of_multiplied = seq_matrix_sum_row(multiplied_vectors, 0);

    MFREE(mean_diff_vector1);
    MFREE(mean_diff_vector2);
    MFREE(multiplied_vectors);

    return sum_of_multiplied/dataset.rec_count;
}