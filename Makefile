.PHONY: all clean compile_main compile_seq compile_omp compile_cuda compile_test

CC := gcc
LDLIBS := -lm
BUILD := ./build
LIB := ./lib
CFLAGS := 
TO_COMPILE = compile_main compile_seq compile_omp

CUDA_ROOT_DIR := /usr/local/cuda

CUDA_LIB_DIR := -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR := -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS := -lcudart

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

CUDA := ./src/cuda/host
CUDA_SRCS := $(wildcard $(CUDA)/*.cu)
CUDA_OBJS := $(patsubst $(CUDA)/%.cu, $(BUILD)/%.o, $(CUDA_SRCS))

CUDA_KERN := ./src/cuda/device
CUDA_KERN_SRCS := $(wildcard $(CUDA_KERN)/*.cu)
CUDA_KERN_OBJS := $(patsubst $(CUDA_KERN)/%.cu, $(BUILD)/%.o, $(CUDA_KERN_SRCS))

TEST := ./test

OBJS = $(MATRIX_OBJS) $(SEQ_OBJS) $(OMP_OBJS)

ifneq ($(shell which nvcc),)
TO_COMPILE += compile_cuda
OBJS += $(CUDA_OBJS) $(CUDA_KERN_OBJS)
LDLIBS += $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
endif

all: $(LIB) $(BUILD) $(TO_COMPILE)
	$(CC) $(CFLAGS) -shared -o $(LIB)/libmolemath.so $(OBJS) -fopenmp $(LDLIBS)

compile_main:
	$(MAKE) -C $(MATRIX)

compile_seq:
	$(MAKE) -C $(SEQ)

compile_omp:
	$(MAKE) -C $(OMP)

compile_cuda:
	$(MAKE) -C $(CUDA_KERN)
	$(MAKE) -C $(CUDA)

compile_test:
	$(MAKE) -C $(TEST) clean all

$(BUILD) $(LIB):
	@mkdir $@

clean:
	@rm -rf $(BUILD) $(LIB)