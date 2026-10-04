/* Driver: load a .csr graph, run PageRank, report timing and the top nodes.
 *
 * Built several times from the same source (see the Makefile) */

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
            "  -k NUM   how many top nodes to print (default 10)\n"
            "  -o FILE  write every rank to FILE, with the run details in a header\n"
            "  -c FILE  append one CSV row of run details to FILE (for benchmarks)\n"
#ifdef PAGERANK_CUDA
            "  -b T,W   rows of up to T in-neighbours get a thread, up to W a warp,\n"
            "           longer a block (default 16,256; a T beyond the longest row\n"
            "           gives every row its own thread)\n"
            "  -s VAL   share of the edges for the CPU, the longest rows first\n"
            "           (default 0.5; 0 leaves every row to the GPU)\n"
#endif
            , prog);
}

/* Checks that argv[i] is an option followed by its value.  An option is a
 * dash and one letter: the switch in main() looks only at the letter, so
 * without this "foo" would pass for -o and "-dx" for -d.
 * Returns 0, or -1 after saying what is wrong. */
static int check_option(int argc, char **argv, int i)
{
    const char *opt = argv[i];

    if (opt[0] != '-' || opt[1] == '\0' || opt[2] != '\0') {
        fprintf(stderr, "%s: unknown option\n", opt);
        usage(argv[0]);
        return -1;
    }
    if (i + 1 >= argc) {
        fprintf(stderr, "%s: missing value\n", opt);
        return -1;
    }
    return 0;
}

#ifdef PAGERANK_CUDA
/* Reads the "T,W" of -b into params.  Returns 0, or -1 if it is not two
 * non-negative integers separated by a comma. */
static int parse_classes(const char *text, pagerank_params *params)
{
    uint64_t t, w;
    char extra;

    if (strchr(text, '-') != NULL ||
        sscanf(text, "%" SCNu64 ",%" SCNu64 "%c", &t, &w, &extra) != 2) {
        return -1;
    }
    params->thread_max = t;
    params->warp_max   = w;
    return 0;
}
#endif

/* "seq" and "omp" are separate rows in the benchmark log on purpose: an
 * OpenMP build restricted to one thread is not the same thing as a build
 * with no OpenMP at all, and the difference shows up in the timings.  The
 * hybrid build has no GPU-only twin, on the contrary: with -s 0 it runs the
 * same code, and never even starts the OpenMP threads. */
#if defined(PAGERANK_CUDA)
#define BUILD_LABEL "hybrid"
#elif defined(_OPENMP)
#define BUILD_LABEL "omp"
#else
#define BUILD_LABEL "seq"
#endif

/* Enough digits to read the value back bit-for-bit, so that two runs can be
 * compared exactly -- the point of the file is diffing CPU against GPU. */
static int rank_digits(void)
{
    return sizeof(rank_t) == sizeof(double) ? 17 : 9;
}

/* Writes every rank, in node order.  Node order rather than sorted by rank so
 * that two files line up line-by-line and can be compared directly; sorting
 * would reorder near-ties differently between implementations and make a diff
 * useless.  Use `sort -k2 -g -r` afterwards to view them by rank. */
static int write_ranks(const char *path, const csr_graph *g, const uint64_t *ids,
                       const rank_t *rank, const pagerank_params *params,
                       const pagerank_stats *stats, double rank_sum, int threads,
                       const char *device)
{
    FILE *f = fopen(path, "w");
    uint64_t v;

    if (f == NULL) {
        fprintf(stderr, "%s: cannot open for writing\n", path);
        return -1;
    }

    fprintf(f, "# pagerank ranks\n");
    fprintf(f, "# nodes            %" PRIu64 "\n", g->n_nodes);
    fprintf(f, "# edges            %" PRIu64 "\n", g->n_edges);
    fprintf(f, "# build            %s\n", BUILD_LABEL);
    fprintf(f, "# precision        %s\n",
            sizeof(rank_t) == sizeof(double) ? "double" : "float");
    fprintf(f, "# threads          %d\n", threads);
    if (device != NULL) {
        fprintf(f, "# device           %s\n", device);
        fprintf(f, "# classes          %" PRIu64 ",%" PRIu64 "\n",
                params->thread_max, params->warp_max);
        fprintf(f, "# cpu_share        %g\n", params->host_share);
        fprintf(f, "# cpu_rows         %" PRIu64 "\n", stats->host_rows);
        fprintf(f, "# cpu_edges        %" PRIu64 "\n", stats->host_edges);
    }
    fprintf(f, "# damping          %g\n", params->damping);
    fprintf(f, "# tolerance        %g\n", params->tolerance);
    fprintf(f, "# iterations       %d\n", stats->iterations);
    fprintf(f, "# converged        %s\n", stats->converged ? "yes" : "no");
    fprintf(f, "# final_L1_change  %.6e\n", stats->error);
    fprintf(f, "# seconds_total    %.6f\n", stats->seconds);
    fprintf(f, "# seconds_per_iter %.6f\n", stats->seconds / (double)stats->iterations);
    fprintf(f, "# rank_sum         %.15f\n", rank_sum);
    fprintf(f, "# columns          %s rank\n", ids != NULL ? "snap_id" : "node_index");

    for (v = 0; v < g->n_nodes; v++) {
        fprintf(f, "%" PRIu64 " %.*g\n",
                ids != NULL ? ids[v] : v, rank_digits(), (double)rank[v]);
    }

    if (fclose(f) != 0) {
        fprintf(stderr, "%s: error while writing\n", path);
        return -1;
    }
    return 0;
}

/* Appends one row per run, so a sweep over graphs and thread counts builds
 * the results table by itself.  The header is written only when the file is
 * created. */
static int append_csv(const char *path, const char *graph, const csr_graph *g,
                      const pagerank_params *params, const pagerank_stats *stats,
                      double rank_sum, int threads)
{
    int is_new = 0;
    FILE *probe = fopen(path, "r");
    FILE *f;
    /* The cpu_share column: the fraction of the edges the CPU works on.  In
     * the hybrid build it is what -s asked for (0 for a GPU-only run); in the
     * CPU builds the CPU does all the work, so it is 1.  Without it a GPU-only
     * run and a hybrid one would look the same in the CSV, since both have
     * build "hybrid". */
#ifdef PAGERANK_CUDA
    const double cpu_share = params->host_share;
#else
    const double cpu_share = 1.0;
#endif

    if (probe == NULL) {
        is_new = 1;
    } else {
        fclose(probe);
    }

    f = fopen(path, "a");
    if (f == NULL) {
        fprintf(stderr, "%s: cannot open for appending\n", path);
        return -1;
    }
    if (is_new) {
        fprintf(f, "graph,nodes,edges,build,precision,threads,damping,tolerance,"
                   "iterations,converged,seconds_total,seconds_per_iter,rank_sum,"
                   "cpu_share\n");
    }
    fprintf(f, "%s,%" PRIu64 ",%" PRIu64 ",%s,%s,%d,%g,%g,%d,%d,%.6f,%.6f,%.15f,%g\n",
            graph, g->n_nodes, g->n_edges, BUILD_LABEL,
            sizeof(rank_t) == sizeof(double) ? "double" : "float",
            threads, params->damping, params->tolerance,
            stats->iterations, stats->converged,
            stats->seconds, stats->seconds / (double)stats->iterations, rank_sum,
            cpu_share);

    if (fclose(f) != 0) {
        fprintf(stderr, "%s: error while writing\n", path);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    csr_graph g;
    pagerank_params params = pagerank_default_params();
    pagerank_stats stats;
    rank_t *rank;   /* address only -- malloc() below creates the array once g.n_nodes is known */
    uint64_t *ids = NULL;
    const char *ids_path = NULL;
    const char *ranks_path = NULL;
    const char *csv_path = NULL;
    top_entry *top;
    int k = 10, n_top = 0, i, threads;
    uint64_t v;
    accum_t total = 0.0;
    const char *device = NULL;   /* the GPU build's description of its device */
#ifdef PAGERANK_CUDA
    char device_buf[256];
#endif

    if (argc < 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    for (i = 2; i < argc; i += 2) {
        const char *val;

        if (check_option(argc, argv, i) != 0) {
            return EXIT_FAILURE;
        }
        val = argv[i + 1];

        switch (argv[i][1]) {
        case 'i': ids_path         = val;                          break;
        case 'd': params.damping   = strtod(val, NULL);            break;
        case 't': params.tolerance = strtod(val, NULL);            break;
        case 'n': params.max_iters = (int)strtol(val, NULL, 10);   break;
        case 'k': k                = (int)strtol(val, NULL, 10);   break;
        case 'o': ranks_path       = val;                          break;
        case 'c': csv_path         = val;                          break;
#ifdef PAGERANK_CUDA
        case 'b':
            if (parse_classes(val, &params) != 0) {
                fprintf(stderr, "-b wants two row lengths, e.g. -b 16,256\n");
                return EXIT_FAILURE;
            }
            break;
        case 's': params.host_share = strtod(val, NULL);           break;
#endif
        default:
            fprintf(stderr, "%s: unknown option\n", argv[i]);
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (params.damping <= 0.0 || params.damping >= 1.0 || k < 1 || params.max_iters < 1) {
        fprintf(stderr, "invalid parameters: need 0 < d < 1, k >= 1, iterations >= 1\n");
        return EXIT_FAILURE;
    }
    if (params.host_share < 0.0 || params.host_share > 1.0) {
        fprintf(stderr, "invalid parameters: the CPU's share must be between 0 and 1\n");
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

    /* heap, not rank_t rank[N]: g.n_nodes is unknown until now and too big for the stack */
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

#ifdef _OPENMP
    threads = omp_get_max_threads();
#else
    threads = 1;
#endif

#ifdef PAGERANK_CUDA
    if (pagerank_device(device_buf, sizeof(device_buf)) != 0) {
        free(rank);
        free(top);
        free(ids);
        csr_free(&g);
        return EXIT_FAILURE;
    }
    device = device_buf;
#endif

    printf("graph      %s\n", argv[1]);
    printf("           %" PRIu64 " nodes, %" PRIu64 " edges\n", g.n_nodes, g.n_edges);
    printf("precision  %s\n", sizeof(rank_t) == sizeof(double) ? "double" : "float");
#if defined(PAGERANK_CUDA)
    printf("device     %s\n", device);
    printf("classes    a thread per row up to %" PRIu64 " in-neighbours, a warp up to %"
           PRIu64 ", a block beyond\n", params.thread_max, params.warp_max);
    if (params.host_share > 0.0) {
        printf("cpu        the longest rows, up to %g%% of the edges, on %d OpenMP threads\n",
               100.0 * params.host_share, threads);
    } else {
        printf("cpu        no rows (-s 0): GPU only\n");
    }
#elif defined(_OPENMP)
    printf("threads    %d (OpenMP)\n", threads);
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
#ifdef PAGERANK_CUDA
    if (stats.host_rows > 0) {
        printf("cpu took   %" PRIu64 " rows of %" PRIu64 " in-neighbours or more,"
               " %.1f%% of the edges\n", stats.host_rows, stats.host_min_len,
               100.0 * (double)stats.host_edges / (double)g.n_edges);
    } else {
        printf("cpu took   no rows\n");
    }
#endif
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

    if (ranks_path != NULL &&
        write_ranks(ranks_path, &g, ids, rank, &params, &stats, (double)total, threads,
                    device) == 0) {
        printf("\nranks written to %s\n", ranks_path);
    }
    if (csv_path != NULL &&
        append_csv(csv_path, argv[1], &g, &params, &stats, (double)total, threads) == 0) {
        printf("run appended to %s\n", csv_path);
    }

    free(rank);
    free(top);
    free(ids);
    csr_free(&g);
    return EXIT_SUCCESS;
}
