.PHONY: all

NVCC := nvcc
NVCC_FLAGS := --compiler-options '-fPIC'
BUILD := ../../../build

DIR_SRCS := $(wildcard ./*.cu)
DIR_OBJS := $(patsubst ./%.cu, $(BUILD)/%.o, $(DIR_SRCS))

all: $(DIR_OBJS)

$(BUILD)/%.o: ./%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@