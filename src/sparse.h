#ifndef SPARSE_H
#define SPARSE_H

#include <stdlib.h>

/* Compressed Sparse Row matrix. row_ptr has length n+1; col_idx/values have
 * length nnz. Row i's entries live in [row_ptr[i], row_ptr[i+1]). */
struct csr_matrix {
    size_t n;
    size_t nnz;
    size_t *row_ptr;
    size_t *col_idx;
    double *values;
};

/* Row-major edge accumulator, used while a topology is being generated.
 *
 * The dense n*n scratch buffer this replaces costs 8 bytes for every possible
 * connection, whether or not an edge is there. This stores only the edges that
 * exist, at 16 bytes each (column index plus weight), so it is smaller
 * whenever fewer than about a third of the possible connections are present.
 * Note that is a constant-factor saving proportional to connectivity, not a
 * change in growth rate: at a fixed connectivity both still scale with n*n.
 *
 * per_row_hint pre-sizes every row so that generation does not have to keep
 * reallocating. Pass the expected number of edges per row (with a little
 * headroom, since a row that outgrows its hint costs a copy), or 0 to grow
 * purely on demand.
 *
 * The semantics deliberately mirror the dense buffer it replaces:
 *   edge_rows_set(er,i,j,w)  ==  W[i*n + j] = w      (overwrites in place)
 *   edge_rows_set(er,i,j,0)  ==  W[i*n + j] = 0.0    (erases the edge)
 *   edge_rows_has(er,i,j)    ==  W[i*n + j] != 0.0
 * and csr_build_from_rows drops exact zeros just as csr_build_from_dense
 * does, so the matrix built either way is identical. */
struct edge_rows {
    size_t   n;
    size_t  *len;
    size_t  *cap;
    size_t **col;
    double **val;
};

int  edge_rows_init(struct edge_rows *er, size_t n, size_t per_row_hint);
void edge_rows_free(struct edge_rows *er);
int  edge_rows_set(struct edge_rows *er, size_t i, size_t j, double w);
/* O(1) push that skips the duplicate check. Use where the caller knows (i,j)
 * has not been written yet, which is every generation path except rewiring.
 * Should a duplicate slip through, csr_build_from_rows keeps the last one
 * written, so the result still matches a dense assignment. */
void edge_rows_append(struct edge_rows *er, size_t i, size_t j, double w);
int  edge_rows_has(const struct edge_rows *er, size_t i, size_t j);
void edge_rows_to_dense(const struct edge_rows *er, double *dense_out);

struct csr_matrix csr_build_from_rows(const struct edge_rows *er);
struct csr_matrix csr_build_from_dense(const double *dense, size_t n);
void   csr_free(struct csr_matrix *m);
void   csr_to_dense(const struct csr_matrix *m, double *dense_out);
double csr_row_dot(const struct csr_matrix *m, size_t row, const double *x);
void   csr_spmv(const struct csr_matrix *m, const double *x, double *y);
void   csr_scale(struct csr_matrix *m, double factor);
double csr_spectral_radius(const struct csr_matrix *m, size_t n);
double csr_spectral_radius_power_iteration(const struct csr_matrix *m, size_t n);

void   csr_zero_neuron(struct csr_matrix *m, size_t idx);
void   csr_restore_neuron(struct csr_matrix *m, size_t idx, const double *row, const double *col);

#endif // SPARSE_H
