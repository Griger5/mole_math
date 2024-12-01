#include "../../include/mole_math/seq_stats_lin_reg.h"

#include "../../include/mole_math/seq_stats_data_properties.h"

LinParam seq_stats_lin_reg_simple(DataSet dataset, size_t dim1, size_t dim2) {
    LinParam result;
    
    double covar_xy = seq_stats_std_covar_popul(dataset, dim1, dim2);
    double var_x = seq_stats_std_var_biased_1d(dataset, dim1);

    double mean_x = seq_stats_mean_arith_data_field(dataset, dim1);
    double mean_y = seq_stats_mean_arith_data_field(dataset, dim2);

    result.slope = covar_xy/var_x;
    result.intercept = mean_y - result.slope * mean_x;

    return result;
}