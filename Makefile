CC     := gcc
CFLAGS := -std=c11 -Wall -Wextra -O2
BUILD  := build

BINS := $(BUILD)/csr_info

all: $(BINS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c src/csr.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/csr_info: $(BUILD)/csr_info.o $(BUILD)/csr.o
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
