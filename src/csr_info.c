/* Loads a .csr file and prints the same statistics as tools/snap_to_csr.py.
 *
 * The two outputs must agree number for number: that is what checks the C
 * loader against the Python writer, i.e. that both sides read the binary
 * layout the same way. */

#include "csr.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    csr_graph g;
    uint64_t *ids = NULL;
    uint64_t v, len;
    uint64_t dangling = 0, no_in = 0;
    uint64_t bin_thread = 0, bin_warp = 0, bin_block = 0;
    uint64_t max_in = 0, max_in_node = 0;
    uint32_t max_out = 0;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <graph.csr> [graph.ids]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (csr_load(argv[1], &g) != 0) {
        return EXIT_FAILURE;
    }

    if (argc == 3) {
        ids = csr_load_ids(argv[2], g.n_nodes);
        if (ids == NULL) {
            csr_free(&g);
            return EXIT_FAILURE;
        }
    }

    /* One pass over the rows collects every figure we report.  The bins are
     * the row-length classes the CUDA kernel will use to pick a granularity:
     * one thread per short row, one warp per medium row, one block for the
     * long tail of the power-law degree distribution. */
    for (v = 0; v < g.n_nodes; v++) {
        len = g.row_ptr[v + 1] - g.row_ptr[v];

        if (len == 0) {
            no_in++;
        } else if (len <= 4) {
            bin_thread++;
        } else if (len <= 32) {
            bin_warp++;
        } else {
            bin_block++;
        }

        if (len > max_in) {
            max_in = len;
            max_in_node = v;
        }
        if (g.out_deg[v] == 0) {
            dangling++;
        }
        if (g.out_deg[v] > max_out) {
            max_out = g.out_deg[v];
        }
    }

    printf("%s\n", argv[1]);
    printf("  nodes                %12" PRIu64 "\n", g.n_nodes);
    printf("  edges                %12" PRIu64 "\n", g.n_edges);
    printf("  dangling (outdeg 0)  %12" PRIu64 "\n", dangling);
    printf("  no in-edges          %12" PRIu64 "\n", no_in);
    printf("  max in / out degree  %12" PRIu64 " / %" PRIu32 "\n", max_in, max_out);
    printf("  row length distribution (in-neighbours per row):\n");
    printf("    1-4  (thread/row) %12" PRIu64 "  (%5.1f%% of rows)\n",
           bin_thread, 100.0 * (double)bin_thread / (double)g.n_nodes);
    printf("    5-32 (warp/row)   %12" PRIu64 "  (%5.1f%% of rows)\n",
           bin_warp, 100.0 * (double)bin_warp / (double)g.n_nodes);
    printf("    >32  (block/row)  %12" PRIu64 "  (%5.1f%% of rows)\n",
           bin_block, 100.0 * (double)bin_block / (double)g.n_nodes);

    if (ids != NULL) {
        printf("  busiest node         index %" PRIu64 ", SNAP id %" PRIu64
               ", %" PRIu64 " in-edges\n", max_in_node, ids[max_in_node], max_in);
    }

    free(ids);
    csr_free(&g);
    return EXIT_SUCCESS;
}
