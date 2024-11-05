.PHONY: all use_cuda

CC := gcc
CFLAGS := -Wall -Wextra -fpic
BUILD := ../../../build

DIR_SRCS := $(wildcard ./*.c)
DIR_OBJS := $(patsubst ./%.c, $(BUILD)/%.o, $(DIR_SRCS))

ifneq ($(shell which nvcc),)
$(shell nvcc ../../cuda/get_sm_count.cu -o get_sm_count)
NUM_OF_SM := $(shell ./get_sm_count)
$(shell rm get_sm_count)
CFLAGS += -DCUDA_SM_COUNT=$(NUM_OF_SM)
endif

all: $(DIR_OBJS)

$(BUILD)/%.o: ./%.c
	$(CC) $(CFLAGS) -c $< -o $@