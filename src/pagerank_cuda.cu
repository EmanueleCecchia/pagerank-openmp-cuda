/* PageRank on the GPU: the power iteration of pagerank.c, with the two loops
 * of every iteration turned into CUDA kernels.
 *
 * Linked in place of pagerank.o (see the Makefile), so main.c drives it
 * unchanged.  The graph goes to the device once, before the clock starts,
 * and the ranks come back once, after it stops; in between every iteration
 * runs on the GPU, and the only data that crosses the bus is the L1 change
 * the host needs to decide whether to stop.
 *
 * The gather adapts its granularity to the length of the rows, which spans
 * orders of magnitude in a power-law graph: short rows get a thread each,
 * medium rows a warp, long rows a whole block.
 *
 * Nothing about the device is assumed: the multiprocessor count, how many
 * blocks each one holds and the free memory are all queried at run time.
 */

#include "pagerank.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Threads per block: a multiple of the warp size */
#define BLOCK 256
#define WARP  32

/* The scalars each iteration accumulates on the device, kept in one array so
 * that a single memset clears them and a single copy brings them back. */
enum { SCALAR_DANGLING, SCALAR_ERROR, N_SCALARS };

/* Granularity classes, in the order their rows are stored. */
enum { CLASS_THREAD, CLASS_WARP, CLASS_BLOCK, N_CLASSES };

/* Every CUDA call returns an error code.  A missing device or a failed
 * launch is not something the program can recover from, so this stops at
 * the first one and says which call it was; a failed allocation is the
 * exception, and pagerank() reports it by returning -1, as on the CPU. */
#define HANDLE_ERROR(call)                                                 \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,  \
                    cudaGetErrorString(err_));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Same clock as pagerank.c, so that the two builds time the same thing. */
static double wall_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double mib(size_t bytes)
{
    return (double)bytes / (1024.0 * 1024.0);
}

/* ---- reductions ---------------------------------------------------------- */

/* Many threads each hold a piece of a sum; these functions add the pieces
 * together.  Three sums go through here: the contributions of the
 * in-neighbours of a node (warp and block kernels), the L1 error and the
 * dangling mass. */

/* Adds up the v of the 32 lanes of a warp; the total ends up in lane 0.
 * At each step every lane adds the v of the lane `offset` places further on,
 * and offset halves: 16, 8, 4, 2, 1.  After these 5 steps lane 0 holds the
 * sum of all 32.  All 32 lanes must call it together. */
__device__ accum_t warp_sum(accum_t v)
{
    int offset;

    for (offset = WARP / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

/* Adds up the v of all the threads of a block; the total ends up in thread 0.
 * Two levels: each warp adds up its 32 values with warp_sum() and writes its
 * total to shared memory, then warp 0 adds up those 8 totals.
 * All threads of the block must call it. */
__device__ accum_t block_sum(accum_t v)
{
    __shared__ accum_t warp_total[BLOCK / WARP];   /* one total per warp, visible to the whole block */
    const int lane = threadIdx.x % WARP;   /* position inside the warp, 0..31 */
    const int warp = threadIdx.x / WARP;   /* which warp of the block, 0..7 */

    v = warp_sum(v);
    if (lane == 0) {
        warp_total[warp] = v;
    }
    /* Barrier: no thread goes on until all the threads of the block are here.
     * Warp 0 must not read warp_total before all 8 warps have written it. */
    __syncthreads();
    if (warp == 0) {
        /* lanes 0-7 take the 8 totals, the other lanes add nothing */
        v = warp_sum(lane < BLOCK / WARP ? warp_total[lane] : 0.0);
    }
    /* Second barrier: gather_block_rows() calls this once per row, and no
     * warp may write the totals of the next row while warp 0 is still
     * reading those of this one. */
    __syncthreads();
    return v;
}

/* Adds up v over the block and adds the total to *out, a single variable
 * shared by the whole grid: what reduction(+) does in OpenMP. */
__device__ void block_sum_add(accum_t v, accum_t *out)
{
    v = block_sum(v);
    if (threadIdx.x == 0) {
        /* Atomic: reads *out, adds v and writes it back as one indivisible
         * step, so that blocks adding at the same moment do not overwrite
         * each other.  One per block, not one per thread: thousands of
         * threads would queue up on the same address. */
        atomicAdd(out, v);
    }
}

/* ---- kernels ------------------------------------------------------------- */

/* First loop of the iteration: how much rank each node sends along each of
 * its outgoing edges, and the mass of the dangling nodes, which have nowhere
 * to send it.
 *
 * Like every kernel here it loops with a grid-wide stride: the grid is sized
 * to the device, not to the graph, and each thread takes v, v + stride,
 * v + 2*stride, ... until it runs past the end. */
__global__ void contrib_kernel(uint64_t n, const uint32_t *__restrict__ out_deg,
                               const rank_t *__restrict__ cur,
                               rank_t *__restrict__ contrib, accum_t *scalars)
{
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    accum_t dangling = 0.0;
    uint64_t v;

    for (v = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; v < n; v += stride) {
        if (out_deg[v] == 0) {
            contrib[v] = (rank_t)0;
            dangling += (accum_t)cur[v];
        } else {
            /* Divided in rank_t, where pagerank.c divides in double: the
             * exact quotient is rounded to a float once either way (out-degrees
             * below 2^24 convert exactly), so the result is the same bit for
             * bit, and the float build stays off the double-precision units,
             * a small fraction of the single-precision ones on consumer GPUs. */
            contrib[v] = cur[v] / (rank_t)out_deg[v];
        }
    }
    block_sum_add(dangling, &scalars[SCALAR_DANGLING]);
}

/* The base every rank starts from, (1-d)/N plus the dangling mass spread
 * over all nodes.  Complete when a gather kernel reads it: contrib_kernel
 * ran to the end first, the kernels being queued on the same stream. */
__device__ double rank_base(uint64_t n, double d, const accum_t *scalars)
{
    return (1.0 - d) / (double)n + d * scalars[SCALAR_DANGLING] / (double)n;
}

/* Second loop, short rows: one thread per row, the direct translation of the
 * OpenMP loop.  The rows to do are rows[0 .. n_rows-1]; rows == NULL means
 * all of them in order, which is what a thread_max beyond the longest row
 * gives: the naive scheme, every row a thread of its own.
 *
 * Cheap only while rows are short: the 32 threads of a warp walk 32 rows at
 * once, so their reads of col_idx land in 32 different places, and the warp
 * moves on only when its longest row is done. */
__global__ void gather_thread_rows(uint64_t n_rows, const uint32_t *__restrict__ rows,
                                   uint64_t n, double d,
                                   const uint64_t *__restrict__ row_ptr,
                                   const uint32_t *__restrict__ col_idx,
                                   const rank_t *__restrict__ contrib,
                                   const rank_t *__restrict__ cur,
                                   rank_t *__restrict__ nxt, accum_t *scalars)
{
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const double base = rank_base(n, d, scalars);
    accum_t error = 0.0;
    uint64_t i, j;

    for (i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_rows; i += stride) {
        const uint64_t v = rows != NULL ? rows[i] : i;
        accum_t sum = 0.0;
        rank_t r;

        for (j = row_ptr[v]; j < row_ptr[v + 1]; j++) {
            sum += (accum_t)contrib[col_idx[j]];
        }
        r = (rank_t)(base + d * sum);
        nxt[v] = r;
        error += fabs((accum_t)r - (accum_t)cur[v]);
    }
    block_sum_add(error, &scalars[SCALAR_ERROR]);
}

/* Medium rows: one warp per row.  Lane k reads elements k, k+32, ... of the
 * row, so the 32 lanes read 32 consecutive entries of col_idx together, in
 * one or two memory transactions instead of 32; then a shuffle reduction
 * adds up their partial sums. */
__global__ void gather_warp_rows(uint64_t n_rows, const uint32_t *__restrict__ rows,
                                 uint64_t n, double d,
                                 const uint64_t *__restrict__ row_ptr,
                                 const uint32_t *__restrict__ col_idx,
                                 const rank_t *__restrict__ contrib,
                                 const rank_t *__restrict__ cur,
                                 rank_t *__restrict__ nxt, accum_t *scalars)
{
    const uint64_t warps = (uint64_t)gridDim.x * (BLOCK / WARP);
    const int lane = threadIdx.x % WARP;
    const double base = rank_base(n, d, scalars);
    accum_t error = 0.0;   /* only lane 0 adds to it */
    uint64_t i, j;

    /* i is the same in every lane of the warp, so all of them reach
     * warp_sum() together, as the shuffles require. */
    for (i = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) / WARP; i < n_rows; i += warps) {
        const uint32_t v = rows[i];
        accum_t sum = 0.0;

        for (j = row_ptr[v] + lane; j < row_ptr[v + 1]; j += WARP) {
            sum += (accum_t)contrib[col_idx[j]];
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            const rank_t r = (rank_t)(base + d * sum);

            nxt[v] = r;
            error += fabs((accum_t)r - (accum_t)cur[v]);
        }
    }
    block_sum_add(error, &scalars[SCALAR_ERROR]);
}

/* Long rows, the hubs of the power law: one block per row, the same scheme
 * as the warp one with BLOCK threads instead of 32. */
__global__ void gather_block_rows(uint64_t n_rows, const uint32_t *__restrict__ rows,
                                  uint64_t n, double d,
                                  const uint64_t *__restrict__ row_ptr,
                                  const uint32_t *__restrict__ col_idx,
                                  const rank_t *__restrict__ contrib,
                                  const rank_t *__restrict__ cur,
                                  rank_t *__restrict__ nxt, accum_t *scalars)
{
    const double base = rank_base(n, d, scalars);
    accum_t error = 0.0;   /* only thread 0 adds to it */
    uint64_t i, j;

    for (i = blockIdx.x; i < n_rows; i += gridDim.x) {
        const uint32_t v = rows[i];
        accum_t sum = 0.0;

        for (j = row_ptr[v] + threadIdx.x; j < row_ptr[v + 1]; j += BLOCK) {
            sum += (accum_t)contrib[col_idx[j]];
        }
        sum = block_sum(sum);
        if (threadIdx.x == 0) {
            const rank_t r = (rank_t)(base + d * sum);

            nxt[v] = r;
            error += fabs((accum_t)r - (accum_t)cur[v]);
        }
    }
    block_sum_add(error, &scalars[SCALAR_ERROR]);
}

/* ---- host side ----------------------------------------------------------- */

/* How many blocks to launch: as many as the device holds resident at once
 * -- blocks per multiprocessor, which the runtime computes from the
 * kernel's registers and shared memory, times the multiprocessors -- but no
 * more than the work needs.  With the grid-stride loops a single such wave
 * covers a graph of any size. */
static int grid_size(const void *kernel, int multiprocessors, uint64_t blocks_needed)
{
    int per_sm = 0;

    HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, BLOCK, 0));
    if ((uint64_t)per_sm * (uint64_t)multiprocessors < blocks_needed) {
        return per_sm * multiprocessors;
    }
    return (int)blocks_needed;
}

static int row_class(const csr_graph *g, const pagerank_params *params, uint64_t v)
{
    const uint64_t len = g->row_ptr[v + 1] - g->row_ptr[v];

    if (len <= params->thread_max) {
        return CLASS_THREAD;
    }
    return len <= params->warp_max ? CLASS_WARP : CLASS_BLOCK;
}

/* Groups the rows by class into one array, first those for a thread, then
 * those for a warp, then those for a block, each group in increasing row
 * order so that neighbouring threads still get neighbouring rows.  The nodes
 * keep their numbers: renumbering them by class would scatter the
 * in-neighbours that the original order keeps close (the locality table of
 * the report).
 *
 * count[c] receives how many rows class c got, *rows the array, left NULL
 * when every row falls in the thread class: there is nothing to group then,
 * and the kernel walks the rows in order.  Returns 0, or -1 if out of memory. */
static int group_rows(const csr_graph *g, const pagerank_params *params,
                      uint32_t **rows, uint64_t count[N_CLASSES])
{
    uint64_t next[N_CLASSES];
    uint64_t v;
    int c;

    *rows = NULL;
    for (c = 0; c < N_CLASSES; c++) {
        count[c] = 0;
    }
    for (v = 0; v < g->n_nodes; v++) {
        count[row_class(g, params, v)]++;
    }
    if (count[CLASS_THREAD] == g->n_nodes) {
        return 0;
    }

    *rows = (uint32_t *)malloc((size_t)g->n_nodes * sizeof(uint32_t));
    if (*rows == NULL) {
        return -1;
    }
    next[CLASS_THREAD] = 0;
    next[CLASS_WARP]   = count[CLASS_THREAD];
    next[CLASS_BLOCK]  = count[CLASS_THREAD] + count[CLASS_WARP];
    for (v = 0; v < g->n_nodes; v++) {
        (*rows)[next[row_class(g, params, v)]++] = (uint32_t)v;
    }
    return 0;
}

/* Also the first CUDA call of the run, so it is here, and not in the timed
 * code, that the driver pays for setting up its context on the device. */
int pagerank_device(char *buf, size_t size)
{
    cudaDeviceProp prop;
    cudaError_t err;
    size_t free_bytes, total_bytes;
    int dev, count = 0;

    err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        fprintf(stderr, "no usable CUDA device: %s\n",
                err != cudaSuccess ? cudaGetErrorString(err) : "none found");
        return -1;
    }
    HANDLE_ERROR(cudaGetDevice(&dev));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));
    HANDLE_ERROR(cudaMemGetInfo(&free_bytes, &total_bytes));

    snprintf(buf, size, "%s, compute capability %d.%d, %d SMs, %.0f MiB (%.0f free)",
             prop.name, prop.major, prop.minor, prop.multiProcessorCount,
             mib(total_bytes), mib(free_bytes));
    return 0;
}

int pagerank(const csr_graph *g, const pagerank_params *params,
             rank_t *rank, pagerank_stats *stats)
{
    const uint64_t n = g->n_nodes;
    const uint64_t m = g->n_edges;
    const double   d = params->damping;
    const size_t   row_ptr_bytes = (size_t)(n + 1) * sizeof(uint64_t);
    const size_t   col_idx_bytes = (size_t)m * sizeof(uint32_t);
    const size_t   out_deg_bytes = (size_t)n * sizeof(uint32_t);
    const size_t   vector_bytes  = (size_t)n * sizeof(rank_t);
    size_t   rows_bytes, needed;
    uint64_t *d_row_ptr = NULL;
    uint32_t *d_col_idx = NULL, *d_out_deg = NULL, *d_rows = NULL;
    uint32_t *rows = NULL;
    rank_t   *d_rank = NULL, *d_next = NULL, *d_contrib = NULL;
    rank_t   *cur, *nxt, *tmp;
    accum_t  *d_scalars = NULL, *h_scalars = NULL;
    const uint32_t *class_rows[N_CLASSES];
    uint64_t count[N_CLASSES];
    int grid[N_CLASSES], grid_contrib;
    cudaStream_t stream = NULL;
    cudaDeviceProp prop;
    size_t free_bytes, total_bytes;
    int dev, iter, converged = 0, result = -1;
    double start, seconds = 0.0;
    accum_t error = 0.0;
    uint64_t v;

    HANDLE_ERROR(cudaGetDevice(&dev));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));

    if (group_rows(g, params, &rows, count) != 0) {
        return -1;
    }
    rows_bytes = rows != NULL ? (size_t)n * sizeof(uint32_t) : 0;
    needed = row_ptr_bytes + col_idx_bytes + out_deg_bytes + rows_bytes
           + 3 * vector_bytes;     /* cur, nxt, contrib */

    /* Everything must fit at once: there is no out-of-core path (yet). */
    HANDLE_ERROR(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (needed > free_bytes) {
        fprintf(stderr, "the graph needs %.1f MiB on %s, only %.1f MiB are free\n",
                mib(needed), prop.name, mib(free_bytes));
        free(rows);
        return -1;
    }

    if (cudaMalloc(&d_row_ptr, row_ptr_bytes)                   != cudaSuccess ||
        cudaMalloc(&d_col_idx, col_idx_bytes > 0 ? col_idx_bytes : 1) != cudaSuccess ||
        cudaMalloc(&d_out_deg, out_deg_bytes)                   != cudaSuccess ||
        (rows_bytes > 0 && cudaMalloc(&d_rows, rows_bytes)      != cudaSuccess) ||
        cudaMalloc(&d_rank,    vector_bytes)                    != cudaSuccess ||
        cudaMalloc(&d_next,    vector_bytes)                    != cudaSuccess ||
        cudaMalloc(&d_contrib, vector_bytes)                    != cudaSuccess ||
        cudaMalloc(&d_scalars, N_SCALARS * sizeof(accum_t))     != cudaSuccess) {
        fprintf(stderr, "out of device memory for %.1f MiB on %s\n", mib(needed), prop.name);
        (void)cudaGetLastError();   /* clear it, so no later check reports it again */
        goto out;
    }
    /* Page-locked, which the asynchronous copy of the scalars requires. */
    HANDLE_ERROR(cudaMallocHost(&h_scalars, N_SCALARS * sizeof(accum_t)));
    HANDLE_ERROR(cudaStreamCreate(&stream));

    /* Start from the uniform distribution 1/N */
    for (v = 0; v < n; v++) {
        rank[v] = (rank_t)(1.0 / (double)n);
    }
    HANDLE_ERROR(cudaMemcpy(d_row_ptr, g->row_ptr, row_ptr_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_col_idx, g->col_idx, col_idx_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_out_deg, g->out_deg, out_deg_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_rank,    rank,       vector_bytes,  cudaMemcpyHostToDevice));
    if (rows_bytes > 0) {
        HANDLE_ERROR(cudaMemcpy(d_rows, rows, rows_bytes, cudaMemcpyHostToDevice));
    }
    /* Each class gets its own kernel, over its own slice of d_rows, and as
     * many blocks as its rows need: a row per thread, per warp, per block. */
    class_rows[CLASS_THREAD] = d_rows;
    class_rows[CLASS_WARP]   = NULL;
    class_rows[CLASS_BLOCK]  = NULL;
    if (d_rows != NULL) {   /* NULL when the thread class has every row */
        class_rows[CLASS_WARP]  = d_rows + count[CLASS_THREAD];
        class_rows[CLASS_BLOCK] = d_rows + count[CLASS_THREAD] + count[CLASS_WARP];
    }
    grid_contrib       = grid_size((const void *)contrib_kernel, prop.multiProcessorCount,
                                   (n + BLOCK - 1) / BLOCK);
    grid[CLASS_THREAD] = grid_size((const void *)gather_thread_rows, prop.multiProcessorCount,
                                   (count[CLASS_THREAD] + BLOCK - 1) / BLOCK);
    grid[CLASS_WARP]   = grid_size((const void *)gather_warp_rows, prop.multiProcessorCount,
                                   (count[CLASS_WARP] + BLOCK / WARP - 1) / (BLOCK / WARP));
    grid[CLASS_BLOCK]  = grid_size((const void *)gather_block_rows, prop.multiProcessorCount,
                                   count[CLASS_BLOCK]);

    /* The two device buffers alternate, by swapping the
     * pointers handed to the kernels. */
    cur = d_rank;
    nxt = d_next;

    start = wall_seconds();
    for (iter = 1; iter <= params->max_iters; iter++) {
        HANDLE_ERROR(cudaMemsetAsync(d_scalars, 0, N_SCALARS * sizeof(accum_t), stream));
        contrib_kernel<<<grid_contrib, BLOCK, 0, stream>>>(n, d_out_deg, cur, d_contrib,
                                                           d_scalars);
        /* An empty class launches nothing: a grid of zero blocks is an error. */
        if (count[CLASS_THREAD] > 0) {
            gather_thread_rows<<<grid[CLASS_THREAD], BLOCK, 0, stream>>>(
                count[CLASS_THREAD], class_rows[CLASS_THREAD], n, d,
                d_row_ptr, d_col_idx, d_contrib, cur, nxt, d_scalars);
        }
        if (count[CLASS_WARP] > 0) {
            gather_warp_rows<<<grid[CLASS_WARP], BLOCK, 0, stream>>>(
                count[CLASS_WARP], class_rows[CLASS_WARP], n, d,
                d_row_ptr, d_col_idx, d_contrib, cur, nxt, d_scalars);
        }
        if (count[CLASS_BLOCK] > 0) {
            gather_block_rows<<<grid[CLASS_BLOCK], BLOCK, 0, stream>>>(
                count[CLASS_BLOCK], class_rows[CLASS_BLOCK], n, d,
                d_row_ptr, d_col_idx, d_contrib, cur, nxt, d_scalars);
        }
        HANDLE_ERROR(cudaGetLastError());

        /* The host waits here for the whole iteration: it needs the error to
         * decide whether to launch the next one. */
        HANDLE_ERROR(cudaMemcpyAsync(h_scalars, d_scalars, N_SCALARS * sizeof(accum_t),
                                   cudaMemcpyDeviceToHost, stream));
        HANDLE_ERROR(cudaStreamSynchronize(stream));
        error = h_scalars[SCALAR_ERROR];

        tmp = cur;
        cur = nxt;
        nxt = tmp;

        if (error < params->tolerance) {
            converged = 1;
            break;
        }
    }
    seconds = wall_seconds() - start;

    HANDLE_ERROR(cudaMemcpy(rank, cur, vector_bytes, cudaMemcpyDeviceToHost));

    if (stats != NULL) {
        stats->seconds    = seconds;
        stats->iterations = converged ? iter : params->max_iters;
        stats->error      = (double)error;
        stats->converged  = converged;
    }
    result = 0;

out:
    /* cudaFree(NULL) is a no-op, like free(NULL). */
    if (stream != NULL) {
        cudaStreamDestroy(stream);
    }
    cudaFreeHost(h_scalars);
    cudaFree(d_scalars);
    cudaFree(d_contrib);
    cudaFree(d_next);
    cudaFree(d_rank);
    cudaFree(d_rows);
    cudaFree(d_out_deg);
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);
    free(rows);
    return result;
}
