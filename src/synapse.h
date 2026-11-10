#ifndef SYNAPSE_H
#define SYNAPSE_H

#include "plasticity.h"
#include "sparse.h"
#include "synapses/simple.h"
#include "synapses/psc_homogeneous.h"
#include "synapses/psc_heterogeneous.h"
#include "synapses/fractional.h"
#include "synapses/fractional_multiexp.h"

/* Which synapse model. Future hot-swappable models (distributional,
 * biorealistic, memcapacitive, ...) are a separate, later effort. */
enum synapse_type {
    SYNAPSE_SIMPLE = 0,
    SYNAPSE_PSC_HOMOGENEOUS,
    SYNAPSE_PSC_HETEROGENEOUS,
    SYNAPSE_FRACTIONAL,
    SYNAPSE_FRACTIONAL_MULTIEXP
};

/* Storage/compute backend. Explicit and user-selectable (see
 * spires_synapse_backend in spires.h) rather than auto-switched, so behavior
 * stays predictable and reproducible.
 *
 * SYNAPSE_SPARSE (CSR) wins at low connectivity (typical reservoir-computing
 * topologies, ~1-20% density). SYNAPSE_DENSE (plain array + BLAS) can win at
 * higher connectivity, since cblas_ddot/cblas_dgemv vectorize over contiguous
 * memory better than CSR's gather-via-col_idx access pattern. */
enum synapse_backend {
    SYNAPSE_SPARSE = 0,
    SYNAPSE_DENSE
};

/* Opaque: internal storage is fully private to the corresponding
 * implementation file under src/synapses/ (mirrors neuron.c's `void *neuron`
 * dispatch). */
struct synapse_matrix {
    enum synapse_type type;
    enum synapse_backend backend;
    size_t n;
    void *data;
};

/* synapse_params layout is documented per type in the corresponding header
 * under src/synapses/; SYNAPSE_SIMPLE ignores it. */
/* rng is used only by synapse types that sample per-connection parameters
 * (currently PSC_HETEROGENEOUS); others ignore it. May be NULL if no such
 * type is in use. */
/* Build straight from the construction-time edge accumulator, so a large
 * reservoir never has to materialise a dense n*n scratch matrix.
 *
 * Only SYNAPSE_SIMPLE on the sparse backend has a direct path today; every
 * other combination expands the edges into a dense buffer and defers to
 * synapse_build_from_dense, which is exactly what construction did before.
 * The dense backend inherently stores n*n, so there is nothing to save there. */
struct synapse_matrix synapse_build_from_rows(const struct edge_rows *er, size_t n,
                                               enum synapse_type type,
                                               enum synapse_backend backend,
                                               const double *synapse_params,
                                               struct spires_rng *rng);

struct synapse_matrix synapse_build_from_dense(const double *dense, size_t n,
                                                enum synapse_type type,
                                                enum synapse_backend backend,
                                                const double *synapse_params,
                                                struct spires_rng *rng);
void   synapse_free(struct synapse_matrix *w);
void   synapse_to_dense(const struct synapse_matrix *w, double *dense_out);
double synapse_row_dot(const struct synapse_matrix *w, size_t row, const double *x);
void   synapse_scale(struct synapse_matrix *w, double factor);
double synapse_spectral_radius(const struct synapse_matrix *w);

/* Apply one plasticity step against the current spike vector. Traces are
 * expected to have been decayed already and are accumulated afterwards, so
 * this sees only strictly-earlier activity. No-op when type is
 * PLASTICITY_NONE. */
void   synapse_apply_plasticity(struct synapse_matrix *w,
                                const double *spikes,
                                struct plasticity_state *ps, double dt);

/* Called once per micro-step, before the per-neuron row_dot pass (it mutates
 * shared per-synapse-model state, so must run single-threaded/first).
 * Returns the vector row_dot should be called with -- for SYNAPSE_SIMPLE this
 * is `spikes` unchanged; for stateful models it's an internally-maintained
 * filtered trace (or unused/NULL, for models like PSC_HETEROGENEOUS that
 * read their own per-edge trace directly inside row_dot instead). */
const double *synapse_prepare(struct synapse_matrix *w, const double *spikes, double dt);

/* Severs a neuron: zeros its afferent row and efferent column in place (the
 * sparsity pattern is preserved, the entries just become explicit zeros). */
void   synapse_zero_neuron(struct synapse_matrix *w, size_t neuron_idx);
/* Inverse of the above: sets neuron_idx's afferent row to `row` and its efferent
 * column to `col` (both length n). Relies on zero_neuron having preserved the
 * sparsity pattern, so entries that were never stored stay absent. row and col
 * must agree on the diagonal (row[neuron_idx] == col[neuron_idx]); the column
 * pass runs second and wins otherwise. Per-synapse-model dynamic state (PSC
 * traces, fractional history) is NOT restored -- see each implementation. */
void   synapse_restore_neuron(struct synapse_matrix *w, size_t neuron_idx,
                              const double *row, const double *col);

#endif // SYNAPSE_H
