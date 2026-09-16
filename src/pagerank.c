/* PageRank by power iteration over the transposed CSR that
 * tools/snap_to_csr.py produces.
 *
 * This builds both the sequential and the OpenMP version: compiled
 * without -fopenmp the compiler ignores the pragmas and emits ordinary
 * serial loops.
 */

#define _POSIX_C_SOURCE 200809L

#include "pagerank.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Current instant in seconds; a duration is the difference of two readings.
 * Elapsed time, not CPU time, which with OpenMP would sum the threads and
 * show no speed-up at all. */
static double wall_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

pagerank_params pagerank_default_params(void)
{
    pagerank_params p;

    p.damping   = 0.85;
    p.tolerance = 1e-6;
    p.max_iters = 100;
    return p;
}

int pagerank(const csr_graph *g, const pagerank_params *params,
             rank_t *rank, pagerank_stats *stats)
{
    const uint64_t n = g->n_nodes;
    const double   d = params->damping;
    rank_t  *next    = malloc((size_t)n * sizeof(rank_t));  // PR', the updated ranks
    rank_t  *contrib = malloc((size_t)n * sizeof(rank_t));  // PR[u] / outdeg(u), one per node
    rank_t  *cur, *nxt;
    uint64_t v;
    int iter, converged = 0;
    double start;
    accum_t error = 0.0;

    if (next == NULL || contrib == NULL) {
        free(next);
        free(contrib);
        return -1;
    }

    /* Start from the uniform distribution 1/N, which already sums to 1. */
    for (v = 0; v < n; v++) {
        rank[v] = (rank_t)(1.0 / (double)n);
    }

    /* The iteration alternates between two buffers: cur holds the current
     * ranks, nxt receives the new ones, and at the end of every pass the two
     * are swapped through a temporary -- three assignments, not n numbers
     * copied. */
    cur = rank;
    nxt = next;

    start = wall_seconds();
    for (iter = 1; iter <= params->max_iters; iter++) {
        accum_t dangling = 0.0;
        rank_t *tmp;
        double base;

        error = 0.0;

        /* How much rank each node sends along each of its outgoing edges. */
#pragma omp parallel for schedule(static) reduction(+ : dangling)
        for (v = 0; v < n; v++) {
            if (g->out_deg[v] == 0) {
                contrib[v] = (rank_t)0;
                dangling += (accum_t)cur[v];
            } else {
                contrib[v] = (rank_t)((accum_t)cur[v] / (accum_t)g->out_deg[v]);
            }
        }

        base = (1.0 - d) / (double)n + d * (double)dangling / (double)n;

        /* Gather from every in-neighbour; dynamic because row lengths are very uneven. */
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : error)
        for (v = 0; v < n; v++) {
            accum_t sum = 0.0;
            uint64_t j;

            /* From row_ptr[v] (included) to row_ptr[v+1] (not included) are
             * the offsets into col_idx of the nodes that point to v; this
             * sums their contributions. */
            for (j = g->row_ptr[v]; j < g->row_ptr[v + 1]; j++) {
                sum += (accum_t)contrib[g->col_idx[j]];
            }
            nxt[v] = (rank_t)(base + d * sum);
            error += fabs((accum_t)nxt[v] - (accum_t)cur[v]); /* fabs: absolute value of a double */
        }

        tmp = cur;
        cur = nxt;
        nxt = tmp;

        if (error < params->tolerance) {
            converged = 1;
            break;
        }
    }

    if (stats != NULL) {
        stats->seconds    = wall_seconds() - start;
        stats->iterations = converged ? iter : params->max_iters;
        stats->error      = (double)error;
        stats->converged  = converged;
    }

    /* An odd number of swaps leaves the result in our scratch buffer. */
    if (cur != rank) {
        memcpy(rank, cur, (size_t)n * sizeof(rank_t));
    }

    free(next);
    free(contrib);
    return 0;
}
