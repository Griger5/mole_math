#include "../../../include/mole_math/stats_lin_reg.h"

#include "../../../include/mole_math/seq_stats_lin_reg.h"

LinParam stats_lin_reg_simple(DataSet dataset, size_t dim1, size_t dim2, char flag) {
    LinParam result;
    
    switch (flag) {
        case 's':
            result = seq_stats_lin_reg_simple(dataset, dim1, dim2);
            break;
        default:
            result = seq_stats_lin_reg_simple(dataset, dim1, dim2);
            break;
    }

    return result;
}