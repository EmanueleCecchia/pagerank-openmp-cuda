CC       := gcc
CFLAGS   := -std=c11 -Wall -Wextra -O2
OMPFLAGS := -fopenmp
LDLIBS   := -lm
BUILD    := build

# pagerank.c carries the OpenMP pragmas.  Compiled without -fopenmp the
# compiler ignores them and emits plain serial loops, so the sequential and
# the parallel binary are built from exactly the same source.
# Ignoring the pragmas is the point of the sequential build, so silence the
# warning that says so; otherwise it fires once per pragma on every build.
SEQFLAGS := -Wno-unknown-pragmas

OBJS := csr.o pagerank.o main.o

SEQ_OBJS := $(addprefix $(BUILD)/seq/,$(OBJS))
OMP_OBJS := $(addprefix $(BUILD)/omp/,$(OBJS))
FLT_OBJS := $(addprefix $(BUILD)/omp-float/,$(OBJS))

all: $(BUILD)/csr_info $(BUILD)/pagerank_seq $(BUILD)/pagerank_omp \
     $(BUILD)/pagerank_omp_float

$(BUILD) $(BUILD)/seq $(BUILD)/omp $(BUILD)/omp-float:
	mkdir -p $@

$(BUILD)/seq/%.o: src/%.c src/csr.h src/pagerank.h | $(BUILD)/seq
	$(CC) $(CFLAGS) $(SEQFLAGS) -c $< -o $@

$(BUILD)/omp/%.o: src/%.c src/csr.h src/pagerank.h | $(BUILD)/omp
	$(CC) $(CFLAGS) $(OMPFLAGS) -c $< -o $@

$(BUILD)/omp-float/%.o: src/%.c src/csr.h src/pagerank.h | $(BUILD)/omp-float
	$(CC) $(CFLAGS) $(OMPFLAGS) -DPAGERANK_FLOAT -c $< -o $@

$(BUILD)/pagerank_seq: $(SEQ_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/pagerank_omp: $(OMP_OBJS)
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/pagerank_omp_float: $(FLT_OBJS)
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/csr_info: $(BUILD)/seq/csr_info.o $(BUILD)/seq/csr.o
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
