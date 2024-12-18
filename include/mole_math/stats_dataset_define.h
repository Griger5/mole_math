#ifndef STATS_DATASET_DEFINE_H
#define STATS_DATASET_DEFINE_H

#define PRIVATE_MAT
#include <mole_math/matrix_define.h>

typedef struct _DataSet {
    struct { 
        size_t rec_count;
        size_t field_count;
        size_t label_count;
        Matrix data;
    } PRIVATE_DATASET;
} DataSet;

size_t stats_dataset_get_record_count(const DataSet dataset);

size_t stats_dataset_get_field_count(const DataSet dataset);

DataSet stats_load_data_csv(const char *filename);

void stats_free_dataset(DataSet dataset);

void stats_print_dataset(DataSet dataset);

#endif