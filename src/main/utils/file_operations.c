#include <stdlib.h>
#include <string.h>

#include "../../../include/mole_math/file_operations.h"

FILE *open_file(const char *filename) {
    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        perror("Unable to open the file");
        exit(1);
    }

    return file;
}

size_t count_records(FILE *file) {
    size_t counter = 0;
    
    fseek(file, 0, SEEK_END);
    size_t length = ftell(file);
    rewind(file);

    char* file_content = malloc(length);
    fread(file_content, 1, length, file);
    rewind(file);

    for (size_t i = 0; i<length; i++) {
        if (file_content[i] == '\n') counter++;
        else if (file_content[i] == '\0' && file_content[i-1] != '\n') counter ++; 
    }

    free(file_content);

    return counter;
}

size_t count_fields(FILE *file) {
    size_t counter = 0;
    char *buffer = NULL;
    size_t n = 0;

    getline(&buffer, &n, file);
    rewind(file);

    char* comma = strchr(buffer, ',');

    while (comma != NULL) {
        counter++;
        comma = strchr(comma+1, ',');
    }

    free(buffer);

    return counter+1;
}