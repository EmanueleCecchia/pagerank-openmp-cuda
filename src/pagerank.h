#ifndef PAGERANK_H
#define PAGERANK_H

#include <stddef.h>

#include "csr.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Precision of the rank and contribution vectors */
#ifdef PAGERANK_FLOAT
typedef float rank_t;
#else
typedef double rank_t;
#endif

/* Scalar accumulator type.  Always double, whatever rank_t is: the dangling
 * mass adds half a million ranks of magnitude ~1/N, and the roundings pile up
 * far enough to matter against a tolerance of 1e-6.  Three scalars cost no
 * memory traffic, so keeping them wide trades away nothing. */
typedef double accum_t;

typedef struct {
    double damping;     /* d in the PageRank formula, conventionally 0.85 */
    double tolerance;   /* stop once the L1 change falls below this */
    int    max_iters;   /* stops the loop if the tolerance is never reached */
    /* hybrid build only, ignored by the others: the granularity classes of
     * the GPU's gather, and the share of the edges whose rows go to the CPU
     * instead, the longest rows first (0 leaves every row to the GPU) */
    uint64_t thread_max;
    uint64_t warp_max;
    double   host_share;
} pagerank_params;

typedef struct {
    int    iterations;  /* number of iterations actually performed */
    double error;       /* L1 change at the last iteration */
    double seconds;     /* wall-clock time of the iteration loop */
    int    converged;   /* non-zero if the tolerance was reached */
    /* hybrid build only: the rows the CPU took -- how many, how many
     * in-neighbours they add up to, and how long the shortest of them is */
    uint64_t host_rows;
    uint64_t host_edges;
    uint64_t host_min_len;
} pagerank_stats;

/* The CPU and the hybrid builds link different implementations of
 * pagerank(), and both must start from the same defaults. */
static inline pagerank_params pagerank_default_params(void)
{
    pagerank_params p;

    p.damping   = 0.85;
    p.tolerance = 1e-6;
    p.max_iters = 100;
    p.thread_max = 16;
    p.warp_max   = 256;
    p.host_share = 0.5; /* Half the edges to the CPU */
    return p;
}

/* Computes the PageRank of g into the rank array, which must have room for
 * g->n_nodes entries.  Returns 0 on success, -1 if working memory could not
 * be allocated (on the hybrid build, device memory too).  stats may be NULL. */
int pagerank(const csr_graph *g, const pagerank_params *params,
             rank_t *rank, pagerank_stats *stats);

#ifdef PAGERANK_CUDA
/* Writes into buf a one-line description of the GPU that pagerank() will
 * use: name, compute capability, multiprocessors and memory, all queried at
 * run time.  Returns 0, or -1 if there is no usable device. */
int pagerank_device(char *buf, size_t size);
#endif

#ifdef __cplusplus
}
#endif

#endif /* PAGERANK_H */
