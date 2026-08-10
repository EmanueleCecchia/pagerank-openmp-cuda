#ifndef PAGERANK_H
#define PAGERANK_H

#include "csr.h"

/* Precision of the rank vector.  Compile with -DPAGERANK_FLOAT for single
 * precision: consumer-grade GPUs commonly run FP64 at a small fraction of
 * their FP32 rate (query with cudaGetDeviceProperties at runtime, per the
 * portability requirement -- never assume a specific device), so the CUDA
 * kernel will likely want float.  Keeping both reachable from one source
 * lets the accuracy and the speed of the two be compared. */
#ifdef PAGERANK_FLOAT
typedef float rank_t;
#else
typedef double rank_t;
#endif

/* Sums are always accumulated in double, whatever rank_t is: adding millions
 * of terms of magnitude ~1/N loses far too much in single precision. */
typedef double accum_t;

typedef struct {
    double damping;     /* d in the PageRank formula, conventionally 0.85 */
    double tolerance;   /* stop once the L1 change falls below this */
    int    max_iters;   /* backstop if the tolerance is never reached */
} pagerank_params;

typedef struct {
    int    iterations;  /* iterations actually performed */
    double error;       /* L1 change at the last iteration */
    double seconds;     /* wall-clock time of the iteration loop */
    int    converged;   /* non-zero if the tolerance was reached */
} pagerank_stats;

pagerank_params pagerank_default_params(void);

/* Computes the PageRank of g into rank, which must have room for g->n_nodes
 * entries.  Returns 0 on success, -1 if working memory could not be
 * allocated.  stats may be NULL. */
int pagerank(const csr_graph *g, const pagerank_params *params,
             rank_t *rank, pagerank_stats *stats);

#endif /* PAGERANK_H */
