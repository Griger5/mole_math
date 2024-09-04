.PHONY: all clean compile_main compile_seq compile_omp compile_test

CC := gcc
LDLIBS := -lm
BUILD := ./build
LIB := ./lib

MAIN := ./src/main

MATRIX = $(MAIN)/matrix
MATRIX_SRCS := $(wildcard $(MATRIX)/*.c)
MATRIX_OBJS := $(patsubst $(MATRIX)/%.c, $(BUILD)/%.o, $(MATRIX_SRCS))

SEQ := ./src/sequential
SEQ_SRCS := $(wildcard $(SEQ)/*.c)
SEQ_OBJS := $(patsubst $(SEQ)/%.c, $(BUILD)/%.o, $(SEQ_SRCS))

OMP := ./src/openmp
OMP_SRCS := $(wildcard $(OMP)/*.c)
OMP_OBJS := $(patsubst $(OMP)/%.c, $(BUILD)/%.o, $(OMP_SRCS))

TEST := ./test

all: $(LIB) $(BUILD) compile_main compile_seq compile_omp
	$(CC) -shared -o $(LIB)/libmolemath.so $(MATRIX_OBJS) $(SEQ_OBJS) $(OMP_OBJS) -fopenmp $(LDLIBS)

compile_main:
	$(MAKE) -C $(MATRIX)

compile_seq:
	$(MAKE) -C $(SEQ)

compile_omp:
	$(MAKE) -C $(OMP)

compile_test:
	$(MAKE) -C $(TEST) clean all

$(BUILD) $(LIB):
	@mkdir $@

clean:
	@rm -rf $(BUILD) $(LIB)