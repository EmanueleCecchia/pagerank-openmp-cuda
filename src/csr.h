#ifndef CSR_H
#define CSR_H

#include <stdint.h>

/* Binary CSR graph, as written by tools/snap_to_csr.py.
 *
 * Row v holds the *in-neighbours* of v, i.e. the transpose of the input
 * graph: PageRank pulls rank from incoming edges, so this is the direction
 * the kernels iterate.  The in-neighbours of v are
 *
 *     col_idx[row_ptr[v]] ... col_idx[row_ptr[v + 1] - 1]
 *
 * and they are sorted in ascending order.  out_deg[u] is the number of
 * outgoing edges of u, needed to divide u's contribution; out_deg[u] == 0
 * marks a dangling node, whose rank must be redistributed over all nodes.
 */
typedef struct {
    uint64_t  n_nodes;   /* N */
    uint64_t  n_edges;   /* M */
    uint64_t *row_ptr;   /* N + 1 offsets into col_idx */
    uint32_t *col_idx;   /* M in-neighbour node indices */
    uint32_t *out_deg;   /* N out-degrees; 0 marks a dangling node */
} csr_graph;

/* Loads the graph at path into g.  Returns 0 on success, -1 on failure
 * (after printing the reason to stderr).  On failure g is left zeroed, so
 * calling csr_free() on it is still safe. */
int csr_load(const char *path, csr_graph *g);

/* Releases the arrays of g and zeroes it.  Safe on an already-freed or
 * never-loaded graph. */
void csr_free(csr_graph *g);

/* Loads the companion .ids file: the original SNAP id of each node index,
 * used only to report results in terms of the ids of the source dataset.
 * Returns a malloc'ed array of n_nodes entries, or NULL on failure. */
uint64_t *csr_load_ids(const char *path, uint64_t n_nodes);

#endif /* CSR_H */
