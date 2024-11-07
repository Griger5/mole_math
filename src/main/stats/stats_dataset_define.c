#include <string.h>
#include <stdlib.h>

#include "../../../include/mole_math/stats_dataset_define.h"
#include "../../../include/mole_math/file_operations.h"
#include "../../../include/mole_math/matrix_utils.h"

void create_datapoint_csv(char *data, DataSet *dataset, size_t index) {
    char *token;
    size_t field_count = dataset->field_count;

    token = strtok(data, ",");
    dataset->data.values[index][0] = atof(token);

    for (size_t i = 1; i < field_count; i++) {
        token = strtok(NULL, ",");
        dataset->data.values[index][i] = atof(token);
    }
}

DataSet stats_load_data_csv(const char *filename) {
    FILE *csv_file = open_file(filename);

    size_t rec_count = count_records(csv_file);
    size_t field_count = count_fields(csv_file);

    DataSet all_data;
    all_data.rec_count = rec_count;
    all_data.field_count = field_count;
    all_data.label_count = 0;
    all_data.data = matrix_init(field_count, rec_count);

    char *current_record = NULL;
    size_t n = 0;
    int chars_num;

    for (size_t i = 0; i < rec_count; i++) {
        chars_num = getline(&current_record, &n, csv_file);
        if (chars_num != -1) {
            create_datapoint_csv(current_record, &all_data, i);
        }
    }

    free(current_record);
    fclose(csv_file);

    return all_data;
}

void stats_free_dataset(DataSet dataset) {
    MFREE(dataset.data);
}

void stats_print_dataset(DataSet dataset) {
    printf("Record count = %ld\n", dataset.rec_count);
    printf("Field count = %ld\n", dataset.field_count);
    matrix_print(dataset.data);
}