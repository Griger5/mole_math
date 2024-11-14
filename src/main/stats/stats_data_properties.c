#include "../../../include/mole_math/stats_data_properties.h"

#include "../../../include/mole_math/seq_stats_data_properties.h"

double stats_sum_data_field(DataSet dataset, size_t field_num, char flag) {
    double result;
    
    switch (flag) {
        case 's':
            result = seq_stats_sum_data_field(dataset, field_num);
            break;
        default:
            result = seq_stats_sum_data_field(dataset, field_num);
            break;
    }

    return result;
}


double stats_mean_arith_data_field(DataSet dataset, size_t field_num, char flag) {
    double result;
    
    switch (flag) {
        case 's':
            result = seq_stats_mean_arith_data_field(dataset, field_num);
            break;
        default:
            result = seq_stats_mean_arith_data_field(dataset, field_num);
            break;
    }

    return result;
}

Matrix stats_mean_diff_data_field(DataSet dataset, size_t field_num, char flag) {
    Matrix result;
    
    switch (flag) {
        case 's':
            result = seq_stats_mean_diff_data_field(dataset, field_num);
            break;
        default:
            result = seq_stats_mean_diff_data_field(dataset, field_num);
            break;
    }

    return result;
}