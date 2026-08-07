/* Driver: load a .csr graph, run PageRank, report timing and the top nodes.
 *
 * Built twice from the same objects (see the Makefile): pagerank_seq without
 * -fopenmp and pagerank_omp with it. */

#include "csr.h"
#include "pagerank.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    uint64_t node;
    rank_t   value;
} top_entry;

/* Keeps the k highest-ranked nodes in descending order.  For the k of
 * interest (10 or so) this beats sorting all N ranks. */
static void top_k_update(top_entry *top, int k, int *n_top, uint64_t node, rank_t value)
{
    int pos;

    if (*n_top == k && value <= top[k - 1].value) {
        return;
    }
    pos = (*n_top < k) ? (*n_top)++ : k - 1;
    while (pos > 0 && top[pos - 1].value < value) {
        top[pos] = top[pos - 1];
        pos--;
    }
    top[pos].node  = node;
    top[pos].value = value;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s <graph.csr> [options]\n"
            "  -i FILE  companion .ids file, to report original SNAP ids\n"
            "  -d VAL   damping factor            (default 0.85)\n"
            "  -t VAL   L1 convergence tolerance  (default 1e-6)\n"
            "  -n NUM   maximum iterations        (default 100)\n"
            "  -k NUM   how many top nodes to print (default 10)\n",
            prog);
}

int main(int argc, char **argv)
{
    csr_graph g;
    pagerank_params params = pagerank_default_params();
    pagerank_stats stats;
    rank_t *rank;
    uint64_t *ids = NULL;
    const char *ids_path = NULL;
    top_entry *top;
    int k = 10, n_top = 0, i;
    uint64_t v;
    accum_t total = 0.0;

    if (argc < 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    for (i = 2; i < argc; i++) {
        if (i + 1 >= argc) {
            fprintf(stderr, "%s: missing value\n", argv[i]);
            return EXIT_FAILURE;
        }
        if (strcmp(argv[i], "-i") == 0) {
            ids_path = argv[++i];
        } else if (strcmp(argv[i], "-d") == 0) {
            params.damping = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "-t") == 0) {
            params.tolerance = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "-n") == 0) {
            params.max_iters = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-k") == 0) {
            k = (int)strtol(argv[++i], NULL, 10);
        } else {
            fprintf(stderr, "%s: unknown option\n", argv[i]);
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (params.damping <= 0.0 || params.damping >= 1.0 || k < 1 || params.max_iters < 1) {
        fprintf(stderr, "invalid parameters: need 0 < d < 1, k >= 1, iterations >= 1\n");
        return EXIT_FAILURE;
    }

    if (csr_load(argv[1], &g) != 0) {
        return EXIT_FAILURE;
    }
    if (ids_path != NULL) {
        ids = csr_load_ids(ids_path, g.n_nodes);
        if (ids == NULL) {
            csr_free(&g);
            return EXIT_FAILURE;
        }
    }

    rank = malloc((size_t)g.n_nodes * sizeof(rank_t));
    top  = malloc((size_t)k * sizeof(top_entry));
    if (rank == NULL || top == NULL) {
        fprintf(stderr, "out of memory for the rank vector\n");
        free(rank);
        free(top);
        free(ids);
        csr_free(&g);
        return EXIT_FAILURE;
    }

    printf("graph      %s\n", argv[1]);
    printf("           %" PRIu64 " nodes, %" PRIu64 " edges\n", g.n_nodes, g.n_edges);
    printf("precision  %s\n", sizeof(rank_t) == sizeof(double) ? "double" : "float");
#ifdef _OPENMP
    printf("threads    %d (OpenMP)\n", omp_get_max_threads());
#else
    printf("threads    1 (sequential build, no OpenMP)\n");
#endif
    printf("damping    %g, tolerance %g, max iterations %d\n",
           params.damping, params.tolerance, params.max_iters);

    if (pagerank(&g, &params, rank, &stats) != 0) {
        fprintf(stderr, "pagerank: out of memory\n");
        free(rank);
        free(top);
        free(ids);
        csr_free(&g);
        return EXIT_FAILURE;
    }

    for (v = 0; v < g.n_nodes; v++) {
        total += (accum_t)rank[v];
        top_k_update(top, k, &n_top, v, rank[v]);
    }

    printf("\n%s after %d iterations (final L1 change %.3e)\n",
           stats.converged ? "converged" : "STOPPED at the iteration limit",
           stats.iterations, stats.error);
    printf("time       %.4f s total, %.4f s per iteration\n",
           stats.seconds, stats.seconds / (double)stats.iterations);
    /* The ranks form a probability distribution, so this must be 1: a
     * deviation means the dangling mass was not redistributed correctly. */
    printf("rank sum   %.12f\n", (double)total);

    printf("\ntop %d nodes:\n", n_top);
    for (i = 0; i < n_top; i++) {
        if (ids != NULL) {
            printf("  %2d. node %9" PRIu64 "  SNAP id %9" PRIu64 "   %.9f\n",
                   i + 1, top[i].node, ids[top[i].node], (double)top[i].value);
        } else {
            printf("  %2d. node %9" PRIu64 "   %.9f\n",
                   i + 1, top[i].node, (double)top[i].value);
        }
    }

    free(rank);
    free(top);
    free(ids);
    csr_free(&g);
    return EXIT_SUCCESS;
}
