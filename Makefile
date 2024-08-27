.PHONY: all clean compile_main compile_seq

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

all: $(LIB) $(BUILD) compile_main compile_seq
	$(CC) -shared -o $(LIB)/libmolemath.so $(MATRIX_OBJS) $(SEQ_OBJS) $(LDLIBS) -I.

compile_main:
	$(MAKE) -C $(MATRIX)

compile_seq:
	$(MAKE) -C $(SEQ)

$(BUILD) $(LIB):
	@mkdir $@

clean:
	@rm -rf $(BUILD) $(LIB)