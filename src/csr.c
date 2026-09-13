#include "csr.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* First eight bytes of a .csr file; must match MAGIC in
 * tools/snap_to_csr.py.  The trailing digits are a format version: bump
 * them whenever the layout changes, so that old binaries reject new files
 * instead of misreading them. */
#define CSR_MAGIC     "PRCSR001"
#define CSR_MAGIC_LEN 8

/* malloc() is allowed to return NULL for a zero-sized request, which we
 * would otherwise mistake for an out-of-memory error; ask for one element
 * instead so that a graph with no edges still loads. */
static void *alloc_array(uint64_t count, size_t size)
{
    if (count == 0) {
        count = 1;
    }
    return malloc((size_t)count * size);
}

/* Size of an open file in bytes, or -1 if it cannot be determined.
 * Leaves the read position at the beginning of the file. */
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

/* fread() wrapper that treats a short read as an error. */
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

/* The offsets must cover exactly the whole edge array: row 0 starts at the
 * beginning, and the extra final entry ends at the edge count. */
static int check_rowptr_span(const csr_graph *g, const char *path)
{
    if (g->row_ptr[0] != 0 || g->row_ptr[g->n_nodes] != g->n_edges) {
        fprintf(stderr, "%s: row_ptr does not span [0, n_edges]\n", path);
        return -1;
    }
    return 0;
}

/* Offsets must never go backwards, otherwise row_ptr[v + 1] - row_ptr[v]
 * would be a negative length and the gather loop would run off the array. */
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

/* Every stored neighbour must name a real node.  This is what lets the kernel
 * index contrib[col_idx[j]] with no bounds check of its own. */
static int check_neighbours_are_existing_nodes(const csr_graph *g, const char *path)
{
    uint64_t j;

    for (j = 0; j < g->n_edges; j++) {
        if (g->col_idx[j] >= g->n_nodes) {
            fprintf(stderr, "%s: column index %" PRIu32 " at position %" PRIu64
                            " is not a valid node\n", path, g->col_idx[j], j);
            return -1;
        }
    }
    return 0;
}

/* The invariants the PageRank loops rely on.  Without them a corrupt file
 * would make the kernels read past the end of col_idx.  Costs one pass over
 * the arrays, negligible next to the many iterations that follow. */
static int csr_validate(const csr_graph *g, const char *path)
{
    if (check_rowptr_span(g, path)                   != 0) return -1;
    if (check_rowptr_monotonic(g, path)              != 0) return -1;
    if (check_neighbours_are_existing_nodes(g, path) != 0) return -1;
    return 0;
}

/* The two header checks below must stay in this order and stay adjacent: the
 * first bounds the counts so that the arithmetic in the second cannot
 * overflow on a corrupt header.  csr_load() enforces the order by chaining
 * them with ||, which short-circuits. */

/* Are the counts even possible for a file this size?  The converter stores
 * node ids as uint32, so a valid file cannot claim more nodes than that, and
 * each edge occupies at least 4 bytes. */
static int check_header_counts(const csr_graph *g, long actual, const char *path)
{
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
 * or corrupt file before anything is allocated from its numbers, and makes the
 * reads in csr_load() unable to come up short.  Only safe once
 * check_header_counts() has bounded the counts. */
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

    /* Every failure below jumps to the single cleanup block at the end, so
     * that the file handle and any partial allocation are released exactly
     * once no matter which check fails. */
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

    /* || short-circuits, so the size check never runs on counts the first
     * check has already rejected -- which is what keeps its arithmetic from
     * overflowing. */
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
