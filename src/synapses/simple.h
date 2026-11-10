#ifndef SYNAPSE_SIMPLE_H
#define SYNAPSE_SIMPLE_H

#include <stdlib.h>
#include "../plasticity.h"
#include "../sparse.h"

/* SYNAPSE_SIMPLE: instantaneous scalar-weight multiply (recurrent_input =
 * sum_j w_ij * spike_j). No params used.
 *
 * Opaque: internal storage (CSR vs. dense) is fully private to simple.c; the
 * backend choice is baked in at construction time (two separate build
 * functions) rather than threaded through every call, so callers/dispatchers
 * never need to know which backend is active. */
struct simple_synapse_data;

struct simple_synapse_data *synapse_simple_build_sparse(const double *dense, size_t n);
/* Same result as build_sparse, without ever materialising the dense matrix. */
struct simple_synapse_data *synapse_simple_build_sparse_rows(const struct edge_rows *er);
struct simple_synapse_data *synapse_simple_build_dense(const double *dense, size_t n);
void   synapse_simple_free(struct simple_synapse_data *d);
void   synapse_simple_to_dense(const struct simple_synapse_data *d, double *dense_out);
double synapse_simple_row_dot(const struct simple_synapse_data *d, size_t row, const double *x);
void   synapse_simple_scale(struct simple_synapse_data *d, double factor);
double synapse_simple_spectral_radius(const struct simple_synapse_data *d);

/* One STDP step over every stored weight. Traces must already be decayed and
 * not yet accumulated, so this sees only strictly-earlier spikes. */
void   synapse_simple_apply_stdp(struct simple_synapse_data *d,
                                 const double *spikes,
                                 struct plasticity_state *ps);

void   synapse_simple_zero_neuron(struct simple_synapse_data *d, size_t idx);
void   synapse_simple_restore_neuron(struct simple_synapse_data *d, size_t idx,
                                     const double *row, const double *col);

#endif // SYNAPSE_SIMPLE_H
