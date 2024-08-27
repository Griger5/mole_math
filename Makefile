.PHONY: all clean compile_main compile_seq

CC := gcc
CFLAGS := -Wall -Wextra -fpic
LIBS := -lm
BUILD := ./build
SO := ./so

MAIN := ./src/main

MATRIX = $(MAIN)/matrix
MATRIX_SRCS := $(wildcard $(MATRIX)/*.c)
MATRIX_OBJS := $(patsubst $(MATRIX)/%.c, $(BUILD)/%.o, $(MATRIX_SRCS))

SEQ := ./src/sequential
SEQ_SRCS := $(wildcard $(SEQ)/*.c)
SEQ_OBJS := $(patsubst $(SEQ)/%.c, $(BUILD)/%.o, $(SEQ_SRCS))

all: $(SO) $(BUILD) compile_main compile_seq
	$(CC) $(CFLAGS) -shared -o $(SO)/libmolemath.so $(MATRIX_OBJS) $(SEQ_OBJS) $(LIBS)

compile_main:
	$(MAKE) -C $(MATRIX)

compile_seq:
	$(MAKE) -C $(SEQ)

$(BUILD) $(SO):
	@mkdir $@

clean:
	@rm -rf $(BUILD) $(SO)