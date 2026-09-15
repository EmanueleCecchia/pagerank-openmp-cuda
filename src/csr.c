#include "csr.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* First eight bytes of a .csr file; must match MAGIC in
 * tools/snap_to_csr.py.  The trailing digits are a format version. */
#define CSR_MAGIC     "PRCSR001"
#define CSR_MAGIC_LEN 8

/* malloc() may return NULL for a zero-sized request, which would look like
 * an out-of-memory error; ask for one element so an empty graph still loads. */
static void *alloc_array(uint64_t count, size_t size)
{
    if (count == 0) {
        count = 1;
    }
    return malloc((size_t)count * size);
}

/* Size of an open file, or -1; leaves the read position at the start. */
static long file_size(FILE *f)
{
    long size;

    if (fseek(f, 0, SEEK_END) != 0) {
        return -1;
    }
    size = ftell(f);
    if (fseek(f, 0, SEEK_SET) != 0) {
        return -1;
    }
    return size;
}

/* Compares how many elements fread() read with how many were asked for: if
 * the file ends early fread() reads fewer and says nothing, whereas this
 * wrapper says so by returning -1. */
static int read_exact(FILE *f, void *dst, uint64_t count, size_t size)
{
    if (count == 0) {
        return 0;
    }
    if (fread(dst, size, (size_t)count, f) != (size_t)count) {
        return -1;
    }
    return 0;
}

/* The two ends of row_ptr: the first one must be 0, the last n_edges. */
static int check_rowptr_span(const csr_graph *g, const char *path)
{
    if (g->row_ptr[0] != 0 || g->row_ptr[g->n_nodes] != g->n_edges) {
        fprintf(stderr, "%s: row_ptr does not span [0, n_edges]\n", path);
        return -1;
    }
    return 0;
}

/* row_ptr offsets must never decrease. */
static int check_rowptr_monotonic(const csr_graph *g, const char *path)
{
    uint64_t v;

    for (v = 0; v < g->n_nodes; v++) {
        if (g->row_ptr[v] > g->row_ptr[v + 1]) {
            fprintf(stderr, "%s: row_ptr decreases at row %" PRIu64 "\n", path, v);
            return -1;
        }
    }
    return 0;
}

/* What lets the kernel index contrib[col_idx[j]] with no bounds check.
 * check that each column index is a valid node,
 * if outside [0, n_nodes - 1] the file is corrupt. */
static int check_neighbours_are_existing_nodes(const csr_graph *g, const char *path)
{
    uint64_t j;

    /* checking every column index because it's not guaranteed to be in order */
    for (j = 0; j < g->n_edges; j++) {
        if (g->col_idx[j] >= g->n_nodes) { // col_idx is uint32, so this cannot go negative
            fprintf(stderr, "%s: column index %" PRIu32 " at position %" PRIu64
                            " is not a valid node\n", path, g->col_idx[j], j);
            return -1;
        }
    }
    return 0;
}

static int csr_validate(const csr_graph *g, const char *path)
{
    if (check_rowptr_span(g, path)                   != 0) return -1;
    if (check_rowptr_monotonic(g, path)              != 0) return -1;
    if (check_neighbours_are_existing_nodes(g, path) != 0) return -1;
    return 0;
}

/* Check the counts in the header against the actual file size. */
static int check_header_counts(const csr_graph *g, long actual, const char *path)
{
    /* a graph with zero nodes makes no sense and we avoid division by zero.
     * a valid file cannot declare more nodes than a uint32 can address, that is about 4.3 billion.
     * file size / how many bytes per edge, gives the maximum number of edges that can be stored in the file:
     * If n_edges is greater than this, it means the header claims to have more edges than can fit in the file, which is invalid. */
    if (g->n_nodes == 0 || g->n_nodes > UINT32_MAX ||
        g->n_edges > (uint64_t)actual / sizeof(uint32_t)) {
        fprintf(stderr, "%s: header claims %" PRIu64 " nodes and %" PRIu64 " edges, "
                        "which cannot fit in %ld bytes\n",
                path, g->n_nodes, g->n_edges, actual);
        return -1;
    }
    return 0;
}

/* Does the file measure exactly what the header implies?  Rejects a truncated
 * file before anything is allocated from its numbers.  Runs only after
 * check_header_counts(), which bounds the counts so this cannot overflow. */
static int check_header_size(const csr_graph *g, long actual, const char *path)
{
    uint64_t expected = (uint64_t)CSR_MAGIC_LEN
                      + 2 * sizeof(uint64_t)
                      + (g->n_nodes + 1) * sizeof(uint64_t)
                      + g->n_edges * sizeof(uint32_t)
                      + g->n_nodes * sizeof(uint32_t);

    if (expected != (uint64_t)actual) {
        fprintf(stderr, "%s: size mismatch (header implies %" PRIu64 " bytes, file has %ld)\n",
                path, expected, actual);
        return -1;
    }
    return 0;
}

int csr_load(const char *path, csr_graph *g)
{
    FILE *f;
    char magic[CSR_MAGIC_LEN];
    uint64_t header[2];
    long actual;

    memset(g, 0, sizeof(*g));

    f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "%s: cannot open file\n", path);
        return -1;
    }

    actual = file_size(f);
    if (actual < 0) {
        fprintf(stderr, "%s: cannot determine file size\n", path);
        goto fail;
    }

    if (fread(magic, 1, CSR_MAGIC_LEN, f) != CSR_MAGIC_LEN ||
        memcmp(magic, CSR_MAGIC, CSR_MAGIC_LEN) != 0) {
        fprintf(stderr, "%s: not a " CSR_MAGIC " file\n", path);
        goto fail;
    }

    if (read_exact(f, header, 2, sizeof(uint64_t)) != 0) {
        fprintf(stderr, "%s: truncated header\n", path);
        goto fail;
    }
    g->n_nodes = header[0];
    g->n_edges = header[1];

    if (check_header_counts(g, actual, path) != 0 ||
        check_header_size(g, actual, path) != 0) {
        goto fail;
    }

    g->row_ptr = alloc_array(g->n_nodes + 1, sizeof(uint64_t));
    g->col_idx = alloc_array(g->n_edges,     sizeof(uint32_t));
    g->out_deg = alloc_array(g->n_nodes,     sizeof(uint32_t));
    if (g->row_ptr == NULL || g->col_idx == NULL || g->out_deg == NULL) {
        fprintf(stderr, "%s: out of memory for %" PRIu64 " nodes and %" PRIu64 " edges\n",
                path, g->n_nodes, g->n_edges);
        goto fail;
    }

    if (read_exact(f, g->row_ptr, g->n_nodes + 1, sizeof(uint64_t)) != 0 ||
        read_exact(f, g->col_idx, g->n_edges,     sizeof(uint32_t)) != 0 ||
        read_exact(f, g->out_deg, g->n_nodes,     sizeof(uint32_t)) != 0) {
        fprintf(stderr, "%s: truncated file\n", path);
        goto fail;
    }

    if (csr_validate(g, path) != 0) {
        goto fail;
    }

    fclose(f);
    return 0;

fail:
    fclose(f);
    csr_free(g);
    return -1;
}

void csr_free(csr_graph *g)
{
    /* free(NULL) is a no-op, so this also works on a partially built graph. */
    free(g->row_ptr);
    free(g->col_idx);
    free(g->out_deg);
    memset(g, 0, sizeof(*g));
}

uint64_t *csr_load_ids(const char *path, uint64_t n_nodes)
{
    FILE *f;
    uint64_t *ids;

    f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "%s: cannot open file\n", path);
        return NULL;
    }

    ids = alloc_array(n_nodes, sizeof(uint64_t));
    if (ids == NULL) {
        fprintf(stderr, "%s: out of memory for %" PRIu64 " ids\n", path, n_nodes);
        fclose(f);
        return NULL;
    }

    if (read_exact(f, ids, n_nodes, sizeof(uint64_t)) != 0) {
        fprintf(stderr, "%s: expected %" PRIu64 " ids\n", path, n_nodes);
        free(ids);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return ids;
}
