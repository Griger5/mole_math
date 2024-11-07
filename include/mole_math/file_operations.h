#ifndef FILE_OPERATIONS_H
#define FILE_OPERATIONS_H

#include <stdio.h>

FILE *open_file(const char *filename);

size_t count_records(FILE *file);
size_t count_fields(FILE *file);

#endif