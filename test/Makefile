.PHONY: all clean

CC := gcc
CFLAGS := -Wall -Wextra -fopenmp
BUILD := ./bin
LDLIBS := -lcunit -lmolemath -lm

DIR_SRCS := $(wildcard ./*.c)
DIR_OBJS := $(patsubst ./%.c, $(BUILD)/%.exe, $(DIR_SRCS))

all: $(BUILD) $(DIR_OBJS)

$(BUILD)/%.exe: ./%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(BUILD):
	@mkdir $(BUILD)

clean:
	@rm -rf $(BUILD)