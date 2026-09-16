#ifndef PAGERANK_H
#define PAGERANK_H

#include "csr.h"

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
} pagerank_params;

typedef struct {
    int    iterations;  /* number of iterations actually performed */
    double error;       /* L1 change at the last iteration */
    double seconds;     /* wall-clock time of the iteration loop */
    int    converged;   /* non-zero if the tolerance was reached */
} pagerank_stats;

pagerank_params pagerank_default_params(void);

/* Computes the PageRank of g into the rank array, which must have room for
 * g->n_nodes entries.  Returns 0 on success, -1 if working memory could not
 * be allocated.  stats may be NULL. */
int pagerank(const csr_graph *g, const pagerank_params *params,
             rank_t *rank, pagerank_stats *stats);

#endif /* PAGERANK_H */
