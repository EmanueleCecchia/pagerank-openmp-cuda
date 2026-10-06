CC       := gcc
CFLAGS   := -std=c11 -Wall -Wextra -O2
OMPFLAGS := -fopenmp
LDLIBS   := -lm
BUILD    := build

# Without -fopenmp the pragmas of pagerank.c are ignored and the loops run
# serially: seq and omp come from the same source.  Ignoring them is the
# point, so the warning about it is silenced.
SEQFLAGS := -Wno-unknown-pragmas

# Hybrid build, needs the CUDA Toolkit.  If nvcc is not installed, `make`
# skips the GPU executables and builds only the CPU ones, instead of failing.
NVCC      ?= nvcc
# GPU architecture to compile for.  native = the GPU of the machine where
# you run make (the one the benchmarks then run on).  For a different GPU,
# name it, e.g.  make CUDA_ARCH=-arch=sm_75  (RTX 2080 Ti)
CUDA_ARCH ?= -arch=native
# nvcc warns on every build that a future release will drop the pre-Turing
# targets, among them the sm_61 of Machine 1; known, and nothing to act on.
CUDA_WARN := -Wno-deprecated-gpu-targets
NVCCFLAGS := -O2 -std=c++17 $(CUDA_ARCH) $(CUDA_WARN) -Xcompiler -Wall,-Wextra
HAVE_NVCC := $(shell command -v $(NVCC) 2>/dev/null)

OBJS := csr.o pagerank.o main.o

SEQ_OBJS := $(addprefix $(BUILD)/seq/,$(OBJS))
OMP_OBJS := $(addprefix $(BUILD)/omp/,$(OBJS))
FLT_OBJS := $(addprefix $(BUILD)/omp-float/,$(OBJS))

# main.c and csr.c are shared; pagerank_cuda.cu replaces pagerank.c, and all
# three are compiled with OpenMP as well, for the CPU's share of the gather.
# No GPU-only executable: with -s 0 the hybrid one is that version.
HYB_OBJS   := $(addprefix $(BUILD)/hybrid/,csr.o main.o pagerank_cuda.o)
HFLT_OBJS  := $(addprefix $(BUILD)/hybrid-float/,csr.o main.o pagerank_cuda.o)
CUDA_BINS  := $(BUILD)/pagerank_hybrid $(BUILD)/pagerank_hybrid_float

all: $(BUILD)/csr_info $(BUILD)/pagerank_seq $(BUILD)/pagerank_omp \
     $(BUILD)/pagerank_omp_float

ifneq ($(HAVE_NVCC),)
all: $(CUDA_BINS)
endif

# Builds the GPU executables or fails saying why, whatever `all` decided.
cuda: $(CUDA_BINS)

$(BUILD) $(BUILD)/seq $(BUILD)/omp $(BUILD)/omp-float $(BUILD)/hybrid $(BUILD)/hybrid-float:
	mkdir -p $@

$(BUILD)/seq/%.o: src/%.c src/csr.h src/pagerank.h | $(BUILD)/seq
	$(CC) $(CFLAGS) $(SEQFLAGS) -c $< -o $@

$(BUILD)/omp/%.o: src/%.c src/csr.h src/pagerank.h | $(BUILD)/omp
	$(CC) $(CFLAGS) $(OMPFLAGS) -c $< -o $@

$(BUILD)/omp-float/%.o: src/%.c src/csr.h src/pagerank.h | $(BUILD)/omp-float
	$(CC) $(CFLAGS) $(OMPFLAGS) -DPAGERANK_FLOAT -c $< -o $@

# Two rules per GPU folder: make takes the one whose source exists, the .c
# for the shared files and the .cu for the GPU implementation.  nvcc hands
# -fopenmp to the host compiler, which builds the OpenMP loop of
# pagerank_cuda.cu.
$(BUILD)/hybrid/%.o: src/%.c src/csr.h src/pagerank.h | $(BUILD)/hybrid
	$(CC) $(CFLAGS) $(OMPFLAGS) -DPAGERANK_CUDA -c $< -o $@

$(BUILD)/hybrid/%.o: src/%.cu src/csr.h src/pagerank.h | $(BUILD)/hybrid
	$(NVCC) $(NVCCFLAGS) -Xcompiler $(OMPFLAGS) -DPAGERANK_CUDA -c $< -o $@

$(BUILD)/hybrid-float/%.o: src/%.c src/csr.h src/pagerank.h | $(BUILD)/hybrid-float
	$(CC) $(CFLAGS) $(OMPFLAGS) -DPAGERANK_CUDA -DPAGERANK_FLOAT -c $< -o $@

$(BUILD)/hybrid-float/%.o: src/%.cu src/csr.h src/pagerank.h | $(BUILD)/hybrid-float
	$(NVCC) $(NVCCFLAGS) -Xcompiler $(OMPFLAGS) -DPAGERANK_CUDA -DPAGERANK_FLOAT -c $< -o $@

$(BUILD)/pagerank_seq: $(SEQ_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/pagerank_omp: $(OMP_OBJS)
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/pagerank_omp_float: $(FLT_OBJS)
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/pagerank_hybrid: $(HYB_OBJS)
	$(NVCC) $(CUDA_ARCH) $(CUDA_WARN) -Xcompiler $(OMPFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/pagerank_hybrid_float: $(HFLT_OBJS)
	$(NVCC) $(CUDA_ARCH) $(CUDA_WARN) -Xcompiler $(OMPFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/csr_info: $(BUILD)/seq/csr_info.o $(BUILD)/seq/csr.o
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all cuda clean
