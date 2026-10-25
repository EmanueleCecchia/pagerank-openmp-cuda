/* PageRank on the GPU and the CPU together, the hybrid version.
 *
 * The two loops of each iteration are CUDA kernels.  The accumulation gives
 * short rows a thread, medium rows a warp and long rows a block; the longest
 * rows, up to params->host_share of the edges, go to the OpenMP threads of
 * the host instead.  With a share of 0 the CPU gets nothing: that is the
 * GPU-only version.  The device's limits are all queried at run time.
 */

#include "pagerank.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Threads per block, a multiple of the warp size */
#define BLOCK 256
#define WARP  32

/* Per-iteration scalars, in one array: one memset clears them, one copy reads them. */
enum { SCALAR_DANGLING, SCALAR_ERROR, N_SCALARS };

/* Row classes, in the order their rows are stored; the last is the CPU's. */
enum { CLASS_THREAD, CLASS_WARP, CLASS_BLOCK, CLASS_HOST, N_CLASSES };


#define HANDLE_ERROR(call)                                                 \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,  \
                    cudaGetErrorString(err_));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

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

/* Sum over the 32 lanes of a warp, by shuffles; the total ends up in lane 0.
 * All 32 lanes must call it. */
__device__ accum_t warp_sum(accum_t v)
{
    int offset;

    for (offset = WARP / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

/* Sum over a block: each warp sums its lanes, then warp 0 sums the warp
 * totals.  The total ends up in thread 0; all threads must call it. */
__device__ accum_t block_sum(accum_t v)
{
    __shared__ accum_t warp_total[BLOCK / WARP];
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;

    v = warp_sum(v);
    if (lane == 0) {
        warp_total[warp] = v;
    }
    /* warp 0 reads the totals only once every warp has written its own */
    __syncthreads();
    if (warp == 0) {
        v = warp_sum(lane < BLOCK / WARP ? warp_total[lane] : 0.0);
    }
    /* called once per row: the next row must not overwrite totals still read */
    __syncthreads();
    return v;
}

/* Block sum added to *out with one atomicAdd per block, not per thread: what
 * reduction(+) does in OpenMP. */
__device__ void block_sum_add(accum_t v, accum_t *out)
{
    v = block_sum(v);
    if (threadIdx.x == 0) {
        atomicAdd(out, v);
    }
}

/* ---- kernels ------------------------------------------------------------- */

/* First loop: each node's rank over its out-degree, and the dangling mass.
 * Like every kernel here, a grid-stride loop: the grid is sized to the
 * device, not to the graph. */
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
            /* In rank_t: the same result as dividing in double, and the float
             * build stays off the double-precision units. */
            contrib[v] = cur[v] / (rank_t)out_deg[v];
        }
    }
    block_sum_add(dangling, &scalars[SCALAR_DANGLING]);
}

/* (1-d)/N plus the dangling mass spread over all nodes; the host uses it too,
 * for the CPU's rows. */
__host__ __device__ double rank_base(uint64_t n, double d, const accum_t *scalars)
{
    return (1.0 - d) / (double)n + d * scalars[SCALAR_DANGLING] / (double)n;
}

/* Second loop, short rows: one thread per row.  rows == NULL means every row
 * in order, which is the naive kernel. */
__global__ void accumulate_thread_rows(uint64_t n_rows, const uint32_t *__restrict__ rows,
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

/* Medium rows: one warp per row.  Lane k reads elements k, k+32, ..., so the
 * warp reads col_idx coalesced. */
__global__ void accumulate_warp_rows(uint64_t n_rows, const uint32_t *__restrict__ rows,
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

    /* i is the same for the whole warp, as warp_sum() requires */
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

/* Long rows: one block per row, as the warp kernel with BLOCK threads. */
__global__ void accumulate_block_rows(uint64_t n_rows, const uint32_t *__restrict__ rows,
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

/* Puts the CPU's ranks, packed in the order of rows, back in their place in nxt. */
__global__ void scatter_host_ranks(uint64_t n_rows, const uint32_t *__restrict__ rows,
                                   const rank_t *__restrict__ ranks,
                                   rank_t *__restrict__ nxt)
{
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint64_t i;

    for (i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_rows; i += stride) {
        nxt[rows[i]] = ranks[i];
    }
}

/* ---- host side ----------------------------------------------------------- */

/* Blocks to launch: as many as the device keeps resident (blocks per SM times
 * SMs), or fewer if the work needs fewer. */
static int grid_size(const void *kernel, int multiprocessors, uint64_t blocks_needed)
{
    int per_sm = 0;

    HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, BLOCK, 0));
    if ((uint64_t)per_sm * (uint64_t)multiprocessors < blocks_needed) {
        return per_sm * multiprocessors;
    }
    return (int)blocks_needed;
}

/* The shortest row the CPU takes: whole lengths from the longest down, while
 * their edges stay within share of the total.  *min_len is longer than any
 * row when the CPU gets nothing; *edges receives the edges it gets.  Returns
 * 0, or -1 if out of memory. */
static int host_threshold(const csr_graph *g, double share,
                          uint64_t *min_len, uint64_t *edges)
{
    const double budget = share * (double)g->n_edges;
    uint64_t *edges_of_len;   /* edges in all the rows of each length */
    uint64_t v, len, max_len = 0, taken = 0;

    for (v = 0; v < g->n_nodes; v++) {
        len = g->row_ptr[v + 1] - g->row_ptr[v];
        if (len > max_len) {
            max_len = len;
        }
    }
    edges_of_len = (uint64_t *)calloc((size_t)max_len + 1, sizeof(uint64_t));
    if (edges_of_len == NULL) {
        return -1;
    }
    for (v = 0; v < g->n_nodes; v++) {
        len = g->row_ptr[v + 1] - g->row_ptr[v];
        edges_of_len[len] += len;
    }

    /* empty rows stay on the GPU whatever the share */
    *min_len = max_len + 1;
    for (len = max_len; len >= 1; len--) {
        if ((double)(taken + edges_of_len[len]) > budget) {
            break;
        }
        taken += edges_of_len[len];
        *min_len = len;
    }
    *edges = taken;
    free(edges_of_len);
    return 0;
}

static int row_class(const csr_graph *g, const pagerank_params *params,
                     uint64_t host_min_len, uint64_t v)
{
    const uint64_t len = g->row_ptr[v + 1] - g->row_ptr[v];

    if (len >= host_min_len) {
        return CLASS_HOST;
    }
    if (len <= params->thread_max) {
        return CLASS_THREAD;
    }
    return len <= params->warp_max ? CLASS_WARP : CLASS_BLOCK;
}

/* Groups the rows by class into one array, each class in increasing row
 * order; the nodes keep their numbers, so the locality of the graph is kept.
 * count[c] receives the rows of class c; *rows stays NULL when every row is
 * in the thread class.  Returns 0, or -1 if out of memory. */
static int group_rows(const csr_graph *g, const pagerank_params *params,
                      uint64_t host_min_len, uint32_t **rows, uint64_t count[N_CLASSES])
{
    uint64_t next[N_CLASSES];
    uint64_t v;
    int c;

    *rows = NULL;
    for (c = 0; c < N_CLASSES; c++) {
        count[c] = 0;
    }
    for (v = 0; v < g->n_nodes; v++) {
        count[row_class(g, params, host_min_len, v)]++;
    }
    if (count[CLASS_THREAD] == g->n_nodes) {
        return 0;
    }

    *rows = (uint32_t *)malloc((size_t)g->n_nodes * sizeof(uint32_t));
    if (*rows == NULL) {
        return -1;
    }
    next[CLASS_THREAD] = 0;
    for (c = 1; c < N_CLASSES; c++) {
        next[c] = next[c - 1] + count[c - 1];
    }
    for (v = 0; v < g->n_nodes; v++) {
        (*rows)[next[row_class(g, params, host_min_len, v)]++] = (uint32_t)v;
    }
    return 0;
}

/* The CPU's rows, on the OpenMP threads, reading the copy of contrib.
 * ranks[i] holds the old rank of rows[i] and receives the new one.  Returns
 * the L1 change over these rows. */
static accum_t accumulate_host_rows(const csr_graph *g, uint64_t n_rows, const uint32_t *rows,
                                    double base, double d, const rank_t *contrib, rank_t *ranks)
{
    accum_t error = 0.0;
    uint64_t i;

    /* dynamic: even the longest rows vary in length by orders of magnitude */
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : error)
    for (i = 0; i < n_rows; i++) {
        const uint32_t v = rows[i];
        const rank_t prev = ranks[i];
        accum_t sum = 0.0;
        uint64_t j;

        for (j = g->row_ptr[v]; j < g->row_ptr[v + 1]; j++) {
            sum += (accum_t)contrib[g->col_idx[j]];
        }
        ranks[i] = (rank_t)(base + d * sum);
        error += fabs((accum_t)ranks[i] - (accum_t)prev);
    }
    return error;
}

/* Describes the GPU.  As the first CUDA call it also pays the start-up of the
 * driver, before the clock starts. */
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
    size_t   rows_bytes, host_bytes, needed;
    uint64_t *d_row_ptr = NULL;
    uint32_t *d_col_idx = NULL, *d_out_deg = NULL, *d_rows = NULL;
    uint32_t *rows = NULL;
    rank_t   *d_rank = NULL, *d_next = NULL, *d_contrib = NULL, *d_host_ranks = NULL;
    rank_t   *h_contrib = NULL, *h_host_ranks = NULL;
    rank_t   *cur, *nxt, *tmp;
    accum_t  *d_scalars = NULL, *h_scalars = NULL;
    const uint32_t *class_rows[N_CLASSES];
    const uint32_t *host_rows = NULL;   /* the CPU's rows, in host memory */
    uint64_t count[N_CLASSES], host_min_len, host_edges;
    int grid[N_CLASSES], grid_contrib, c;
    cudaStream_t stream = NULL, copy_stream = NULL;
    cudaEvent_t contrib_done = NULL, contrib_copied = NULL;
    cudaDeviceProp prop;
    size_t free_bytes, total_bytes;
    int dev, iter, converged = 0, result = -1;
    double start, seconds = 0.0;
    accum_t error = 0.0;
    uint64_t v, i;

    HANDLE_ERROR(cudaGetDevice(&dev));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));

    if (host_threshold(g, params->host_share, &host_min_len, &host_edges) != 0 ||
        group_rows(g, params, host_min_len, &rows, count) != 0) {
        return -1;
    }
    rows_bytes = rows != NULL ? (size_t)n * sizeof(uint32_t) : 0;
    host_bytes = (size_t)count[CLASS_HOST] * sizeof(rank_t);
    needed = row_ptr_bytes + col_idx_bytes + out_deg_bytes + rows_bytes
           + 3 * vector_bytes      /* cur, nxt, contrib */
           + host_bytes;           /* the ranks of the CPU's rows */

    /* Everything must fit at once: there is no out-of-core path. */
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
        (host_bytes > 0 && cudaMalloc(&d_host_ranks, host_bytes) != cudaSuccess) ||
        cudaMalloc(&d_scalars, N_SCALARS * sizeof(accum_t))     != cudaSuccess) {
        fprintf(stderr, "out of device memory for %.1f MiB on %s\n", mib(needed), prop.name);
        (void)cudaGetLastError();   /* clear the error */
        goto out;
    }
    /* Page-locked, as asynchronous copies require. */
    HANDLE_ERROR(cudaMallocHost(&h_scalars, N_SCALARS * sizeof(accum_t)));
    HANDLE_ERROR(cudaStreamCreate(&stream));
    if (count[CLASS_HOST] > 0) {
        /* Only when the CPU has rows: page-locked buffers for contrib and the
         * CPU's ranks, a stream for the copy, and two events used only to
         * synchronize. */
        HANDLE_ERROR(cudaMallocHost(&h_contrib,    vector_bytes));
        HANDLE_ERROR(cudaMallocHost(&h_host_ranks, host_bytes));
        HANDLE_ERROR(cudaStreamCreate(&copy_stream));
        HANDLE_ERROR(cudaEventCreateWithFlags(&contrib_done,   cudaEventDisableTiming));
        HANDLE_ERROR(cudaEventCreateWithFlags(&contrib_copied, cudaEventDisableTiming));
    }

    /* Start from the uniform distribution 1/N, the CPU's rows as well */
    for (v = 0; v < n; v++) {
        rank[v] = (rank_t)(1.0 / (double)n);
    }
    for (i = 0; i < count[CLASS_HOST]; i++) {
        h_host_ranks[i] = (rank_t)(1.0 / (double)n);
    }
    HANDLE_ERROR(cudaMemcpy(d_row_ptr, g->row_ptr, row_ptr_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_col_idx, g->col_idx, col_idx_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_out_deg, g->out_deg, out_deg_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_rank,    rank,       vector_bytes,  cudaMemcpyHostToDevice));
    if (rows_bytes > 0) {
        HANDLE_ERROR(cudaMemcpy(d_rows, rows, rows_bytes, cudaMemcpyHostToDevice));
    }
    /* One slice of d_rows per class; the CPU's slice only serves to put its
     * ranks back in place. */
    for (c = 0; c < N_CLASSES; c++) {
        class_rows[c] = NULL;
    }
    if (d_rows != NULL) {   /* NULL when the thread class has every row */
        class_rows[CLASS_THREAD] = d_rows;
        for (c = 1; c < N_CLASSES; c++) {
            class_rows[c] = class_rows[c - 1] + count[c - 1];
        }
        host_rows = rows + (class_rows[CLASS_HOST] - d_rows);
    }
    grid_contrib       = grid_size((const void *)contrib_kernel, prop.multiProcessorCount,
                                   (n + BLOCK - 1) / BLOCK);
    grid[CLASS_THREAD] = grid_size((const void *)accumulate_thread_rows, prop.multiProcessorCount,
                                   (count[CLASS_THREAD] + BLOCK - 1) / BLOCK);
    grid[CLASS_WARP]   = grid_size((const void *)accumulate_warp_rows, prop.multiProcessorCount,
                                   (count[CLASS_WARP] + BLOCK / WARP - 1) / (BLOCK / WARP));
    grid[CLASS_BLOCK]  = grid_size((const void *)accumulate_block_rows, prop.multiProcessorCount,
                                   count[CLASS_BLOCK]);
    grid[CLASS_HOST]   = grid_size((const void *)scatter_host_ranks, prop.multiProcessorCount,
                                   (count[CLASS_HOST] + BLOCK - 1) / BLOCK);

    /* cur and nxt swap at every iteration */
    cur = d_rank;
    nxt = d_next;

    start = wall_seconds();
    for (iter = 1; iter <= params->max_iters; iter++) {
        accum_t host_error = 0.0;

        HANDLE_ERROR(cudaMemsetAsync(d_scalars, 0, N_SCALARS * sizeof(accum_t), stream));
        contrib_kernel<<<grid_contrib, BLOCK, 0, stream>>>(n, d_out_deg, cur, d_contrib,
                                                           d_scalars);
        if (count[CLASS_HOST] > 0) {
            /* contrib and the dangling mass go to the host on the copy stream,
             * which waits for contrib_kernel alone: the copy overlaps the
             * accumulate kernels. */
            HANDLE_ERROR(cudaEventRecord(contrib_done, stream));
            HANDLE_ERROR(cudaStreamWaitEvent(copy_stream, contrib_done, 0));
            HANDLE_ERROR(cudaMemcpyAsync(h_contrib, d_contrib, vector_bytes,
                                         cudaMemcpyDeviceToHost, copy_stream));
            HANDLE_ERROR(cudaMemcpyAsync(&h_scalars[SCALAR_DANGLING], &d_scalars[SCALAR_DANGLING],
                                         sizeof(accum_t), cudaMemcpyDeviceToHost, copy_stream));
            HANDLE_ERROR(cudaEventRecord(contrib_copied, copy_stream));
        }
        /* An empty class launches nothing: a grid of zero blocks is an error. */
        if (count[CLASS_THREAD] > 0) {
            accumulate_thread_rows<<<grid[CLASS_THREAD], BLOCK, 0, stream>>>(
                count[CLASS_THREAD], class_rows[CLASS_THREAD], n, d,
                d_row_ptr, d_col_idx, d_contrib, cur, nxt, d_scalars);
        }
        if (count[CLASS_WARP] > 0) {
            accumulate_warp_rows<<<grid[CLASS_WARP], BLOCK, 0, stream>>>(
                count[CLASS_WARP], class_rows[CLASS_WARP], n, d,
                d_row_ptr, d_col_idx, d_contrib, cur, nxt, d_scalars);
        }
        if (count[CLASS_BLOCK] > 0) {
            accumulate_block_rows<<<grid[CLASS_BLOCK], BLOCK, 0, stream>>>(
                count[CLASS_BLOCK], class_rows[CLASS_BLOCK], n, d,
                d_row_ptr, d_col_idx, d_contrib, cur, nxt, d_scalars);
        }
        HANDLE_ERROR(cudaGetLastError());

        if (count[CLASS_HOST] > 0) {
            /* The CPU's rows, as soon as contrib arrives; their ranks go back
             * behind the accumulate kernels, before the next contrib_kernel. */
            HANDLE_ERROR(cudaEventSynchronize(contrib_copied));
            host_error = accumulate_host_rows(g, count[CLASS_HOST], host_rows,
                                              rank_base(n, d, h_scalars), d,
                                              h_contrib, h_host_ranks);
            HANDLE_ERROR(cudaMemcpyAsync(d_host_ranks, h_host_ranks, host_bytes,
                                         cudaMemcpyHostToDevice, stream));
            scatter_host_ranks<<<grid[CLASS_HOST], BLOCK, 0, stream>>>(
                count[CLASS_HOST], class_rows[CLASS_HOST], d_host_ranks, nxt);
            HANDLE_ERROR(cudaGetLastError());
        }

        /* The host waits for the whole iteration: it needs the error, and
         * h_host_ranks must be copied before it is overwritten. */
        HANDLE_ERROR(cudaMemcpyAsync(h_scalars, d_scalars, N_SCALARS * sizeof(accum_t),
                                   cudaMemcpyDeviceToHost, stream));
        HANDLE_ERROR(cudaStreamSynchronize(stream));
        /* The L1 change over all nodes: the GPU's rows plus the CPU's. */
        error = h_scalars[SCALAR_ERROR] + host_error;

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
        stats->seconds      = seconds;
        stats->iterations   = converged ? iter : params->max_iters;
        stats->error        = (double)error;
        stats->converged    = converged;
        stats->host_rows    = count[CLASS_HOST];
        stats->host_edges   = host_edges;
        stats->host_min_len = host_min_len;
    }
    result = 0;

out:
    /* cudaFree(NULL) is a no-op, like free(NULL). */
    if (contrib_copied != NULL) {
        cudaEventDestroy(contrib_copied);
    }
    if (contrib_done != NULL) {
        cudaEventDestroy(contrib_done);
    }
    if (copy_stream != NULL) {
        cudaStreamDestroy(copy_stream);
    }
    if (stream != NULL) {
        cudaStreamDestroy(stream);
    }
    cudaFreeHost(h_host_ranks);
    cudaFreeHost(h_contrib);
    cudaFreeHost(h_scalars);
    cudaFree(d_scalars);
    cudaFree(d_host_ranks);
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
