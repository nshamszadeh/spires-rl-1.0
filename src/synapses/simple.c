#include "simple.h"
#include "../sparse.h"
#include "../math_utils.h"
#include <string.h>
#include <cblas.h>
#include <math.h>

struct simple_synapse_data {
    int is_sparse;
    size_t n;
    union {
        struct csr_matrix csr;
        double *dense;
    } storage;
};

struct simple_synapse_data *synapse_simple_build_sparse_rows(const struct edge_rows *er)
{
    struct simple_synapse_data *d = malloc(sizeof(*d));
    if (!d)
        return NULL;
    d->is_sparse = 1;
    d->n = er->n;
    d->storage.csr = csr_build_from_rows(er);
    return d;
}

struct simple_synapse_data *synapse_simple_build_sparse(const double *dense, size_t n)
{
    struct simple_synapse_data *d = malloc(sizeof(*d));
    if (!d)
        return NULL;
    d->is_sparse = 1;
    d->n = n;
    d->storage.csr = csr_build_from_dense(dense, n);
    return d;
}

struct simple_synapse_data *synapse_simple_build_dense(const double *dense, size_t n)
{
    struct simple_synapse_data *d = malloc(sizeof(*d));
    if (!d)
        return NULL;
    d->is_sparse = 0;
    d->n = n;
    d->storage.dense = malloc(n * n * sizeof(double));
    if (d->storage.dense)
        memcpy(d->storage.dense, dense, n * n * sizeof(double));
    return d;
}

void synapse_simple_free(struct simple_synapse_data *d)
{
    if (!d)
        return;
    if (d->is_sparse)
        csr_free(&d->storage.csr);
    else
        free(d->storage.dense);
    free(d);
}

void synapse_simple_to_dense(const struct simple_synapse_data *d, double *dense_out)
{
    if (d->is_sparse)
        csr_to_dense(&d->storage.csr, dense_out);
    else
        memcpy(dense_out, d->storage.dense, d->n * d->n * sizeof(double));
}

double synapse_simple_row_dot(const struct simple_synapse_data *d, size_t row, const double *x)
{
    if (d->is_sparse)
        return csr_row_dot(&d->storage.csr, row, x);
    return cblas_ddot((int)d->n, &d->storage.dense[row * d->n], 1, x, 1);
}

void synapse_simple_scale(struct simple_synapse_data *d, double factor)
{
    if (d->is_sparse)
        csr_scale(&d->storage.csr, factor);
    else
        cblas_dscal((int)(d->n * d->n), factor, d->storage.dense, 1);
}

double synapse_simple_spectral_radius(const struct simple_synapse_data *d)
{
    if (d->is_sparse)
        return csr_spectral_radius(&d->storage.csr, d->n);
    return calc_spectral_radius(d->storage.dense, d->n);
}

/* w^mu without calling pow() for the two exponents that actually get used:
 * mu = 1 is linear (fully multiplicative), mu = 0 is constant (additive). */
static inline double wdep(double base, double mu)
{
    if (base <= 0.0) return 0.0;
    if (mu == 1.0)   return base;
    if (mu == 0.0)   return 1.0;
    return pow(base, mu);
}

/* Update one weight. j is the source neuron (column), i the target (row);
 * see test_convention.c for the empirical check of that convention.
 *
 * The rule follows the SOURCE neuron's identity, because that is what the
 * experimental literature separates: excitatory synapses show an asymmetric
 * window (Gutig et al. 2003), inhibitory ones a symmetric window with
 * constant depression (Vogels et al. 2011; D'amour & Froemke 2015).
 *
 * Identity comes from neuron_sign, never from the sign of w: a weight driven
 * to zero has no sign left to read. neuron_sign is therefore required here,
 * which is why plasticity is refused under EI_PER_SYNAPSE.
 *
 * Both branches preserve sign. The excitatory branch is weight-dependent, so
 * depression vanishes as w approaches zero and the weight can never cross it;
 * the inhibitory branch is additive in magnitude and so needs the explicit
 * floor. */
static inline double stdp_one(double w, size_t i, size_t j,
                              const double *spikes,
                              const struct plasticity_state *ps,
                              const double *P)
{
    int src_excitatory = ps->neuron_sign[j] > 0;
    double wmax = P[PL_W_MAX];
    if (wmax <= 0.0) return w;

    if (src_excitatory) {
        double lambda = P[PL_LAMBDA];
        if (spikes[i] != 0.0)                    /* target fired: potentiate */
            w += lambda * wdep(1.0 - w / wmax, P[PL_MU_PLUS]) * ps->x_pre_e[j];
        if (spikes[j] != 0.0)                    /* source fired: depress */
            w -= lambda * P[PL_ALPHA_E] * wdep(w / wmax, P[PL_MU_MINUS]) * ps->x_post_e[i];
        if (w < 0.0)    w = 0.0;
        if (w > wmax)   w = wmax;
        return w;
    }

    /* Symmetric window on the magnitude: either order potentiates inhibition.
     * alpha = 2*tau*rho_0 is the constant depression that sets the target
     * postsynaptic rate, so this branch is homeostatic. */
    double alpha_i = 2.0 * P[PL_TAU_I] * P[PL_RHO_0];
    double eta = P[PL_ETA];
    double mag = -w;
    if (spikes[i] != 0.0)
        mag += eta * ps->x_pre_i[j];
    if (spikes[j] != 0.0)
        mag += eta * (ps->x_post_i[i] - alpha_i);
    if (mag < 0.0)   mag = 0.0;
    if (mag > wmax)  mag = wmax;
    return -mag;
}

void synapse_simple_apply_stdp(struct simple_synapse_data *d,
                               const double *spikes,
                               struct plasticity_state *ps)
{
    if (!d || !ps || !spikes || !ps->params || !ps->neuron_sign)
        return;
    const double *P = ps->params;
    size_t n = d->n;

    if (d->is_sparse) {
        struct csr_matrix *m = &d->storage.csr;
        for (size_t i = 0; i < n; i++) {
            for (size_t k = m->row_ptr[i]; k < m->row_ptr[i + 1]; k++)
                m->values[k] = stdp_one(m->values[k], i, m->col_idx[k], spikes, ps, P);
        }
    } else {
        double *W = d->storage.dense;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                double w = W[i * n + j];
                if (w != 0.0)
                    W[i * n + j] = stdp_one(w, i, j, spikes, ps, P);
            }
        }
    }
}

void synapse_simple_zero_neuron(struct simple_synapse_data *d, size_t idx)
{
    if (!d || idx >= d->n)
        return;
    if (d->is_sparse) {
        csr_zero_neuron(&d->storage.csr, idx);
    } else {
        double *W = d->storage.dense;
        memset(&W[idx * d->n], 0, d->n * sizeof(double));   /* afferent row */
        for (size_t i = 0; i < d->n; i++)                   /* efferent column */
            W[i * d->n + idx] = 0.0;
    }
}

void synapse_simple_restore_neuron(struct simple_synapse_data *d, size_t idx,
                                   const double *row, const double *col)
{
    if (!d || idx >= d->n || !row || !col)
        return;
    if (d->is_sparse) {
        csr_restore_neuron(&d->storage.csr, idx, row, col);
    } else {
        double *W = d->storage.dense;
        memcpy(&W[idx * d->n], row, d->n * sizeof(double)); /* afferent row */
        for (size_t i = 0; i < d->n; i++)                   /* efferent column */
            W[i * d->n + idx] = col[i];
    }
}
