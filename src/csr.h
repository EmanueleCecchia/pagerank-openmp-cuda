#ifndef CSR_H
#define CSR_H

#include <stdint.h>

/* Binary CSR graph, as written by tools/snap_to_csr.py.
 * The graph is stored transposed: row v holds the in-neighbours of v. */
typedef struct {
    uint64_t  n_nodes;   // N
    uint64_t  n_edges;   // M
    uint64_t *row_ptr;   // N + 1
    uint32_t *col_idx;   // M
    uint32_t *out_deg;   // N; 0 marks a dangling node
} csr_graph;

/* Loads the graph (.csr) at path into g.  Returns 0, or -1 after printing
 * the reason to stderr; g is left zeroed on failure, so the caller can
 * csr_free() it unconditionally. */
int csr_load(const char *path, csr_graph *g);

/* Frees the arrays of g and zeroes it, so calling it twice is harmless. */
void csr_free(csr_graph *g);

/* Loads the companion .ids file, the original SNAP id of each node index.
 * Returns a malloc'ed array of n_nodes entries for the caller to free,
 * or NULL after printing the reason to stderr. */
uint64_t *csr_load_ids(const char *path, uint64_t n_nodes);

#endif /* CSR_H */
