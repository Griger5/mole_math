.PHONY: all

CC := gcc
CFLAGS := -Wall -Wextra -fpic
BUILD := ../../../build

DIR_SRCS := $(wildcard ./*.c)
DIR_OBJS := $(patsubst ./%.c, $(BUILD)/%.o, $(DIR_SRCS))

all: $(DIR_OBJS)

$(BUILD)/%.o: ./%.c
	$(CC) $(CFLAGS) -c $< -o $@