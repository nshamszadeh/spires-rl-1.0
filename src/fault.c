#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "fault.h"

struct fault_system* create_fault_system(struct reservoir *r, enum fault_type fault_type, unsigned int rng_state, double neuron_fault_state)
{
    struct fault_system *fs = malloc(sizeof(struct fault_system));
    if (fs == NULL) {
        fprintf(stderr, "Error allocating memory for fault_system.\n");
        return NULL;
    }

    fs->fault_type = fault_type; // set the fault type
    fs->neuron_rng_state = rng_state; // set random neuron index seed
    fs->neuron_fault_state = neuron_fault_state; // neuron fault state
    fs->num_living = r->num_neurons; // number of living (non-faulted) neurons
    
    // save weight matrix pre-fault (dense snapshot; malloc'd by the reservoir)
	fs->W_restore = copy_reservoir_weights(r);
	if ((fs->W_restore) == NULL) {
		fprintf(stderr, "Error allocating memory for W_restore.\n");
        free(fs);
		return NULL;
	}

	// allocate memory for saved neuron information
	fs->living_neurons = malloc(r->num_neurons * sizeof(size_t));
	if ((fs->living_neurons) == NULL) {
		fprintf(stderr, "Error allocating memory for living_neurons.\n");
        free(fs->W_restore);
        free(fs);
		return NULL;
	}
    for (size_t i = 0; i < r->num_neurons; i++) {fs->living_neurons[i] = i;}
	fs->saved_neuron_states = calloc(r->num_neurons, sizeof(struct neuron_state));
	if ((fs->saved_neuron_states) == NULL) {
		fprintf(stderr, "Error allocating memory for saved_neuron_states.\n");
		free(fs->W_restore);
        free(fs->living_neurons);
        free(fs);
        return NULL;
	}
    return fs;
}

void free_fault_system(struct fault_system *fs)
{
    if (!fs)
        return;
    free(fs->W_restore);
    free(fs->saved_neuron_states);
    free(fs->living_neurons);
    fs->W_restore = NULL;
    fs->saved_neuron_states = NULL;
    fs->living_neurons = NULL;
    free(fs);
    fs = NULL;
}

// Returns index of random non-faulty neuron
size_t get_victim_idx(struct fault_system *fs)
{   
    size_t i = (size_t)rand_r(&fs->neuron_rng_state) / (RAND_MAX / fs->num_living + 1);
    size_t neuron_idx = fs->living_neurons[i];
    // update list of living neurons
    // swap victim with furthest living neuron in array
    // decrement number of living neurons after swap
    size_t temp = neuron_idx;
    fs->living_neurons[i] = fs->living_neurons[fs->num_living-1];
    fs->living_neurons[fs->num_living-1] = temp;
    fs->num_living -= 1;
    return neuron_idx;
}

int generate_fault(struct reservoir *r, struct fault_system *fs)
{
    // handled before get_victim_idx so a NONE run doesn't consume a living
    // neuron without marking it failed (that would desync the bookkeeping below)
    if (fs->fault_type == NONE)
        return FAULT_OK;
    if (fs->num_living == 0) {
        fprintf(stdout, "No functioning neurons exist in reservoir!\n");
        return FAULT_EXHAUSTED;
    }
#ifdef DEBUG
    size_t n_dead = 0;
    for (size_t i = 0; i < r->num_neurons; i++) {n_dead += r->failed_neurons[i];}
    if (fs->num_living != r->num_neurons - n_dead) {
        fprintf(stderr, 
            "Error: fault_system and reservoir neuron bookeeping out of sync:\nfs->num_living = %lu, reservoir num_living = %lu\n",
            fs->num_living, r->num_neurons - n_dead);
            fflush(stderr);
            return FAULT_ERROR;
    }
#endif // DEBUG
    // pick a random functioning neuron in the reservoir
    size_t neuron_idx = get_victim_idx(fs);
    switch(fs->fault_type) {
        case CLAMP_LOW:
            r->failed_neurons[neuron_idx] = 1;
            return process_clamp_low(r, fs, neuron_idx);
        default:
            fprintf(stderr, "Error: unknown fault type: %d\n", (int)fs->fault_type);
            return FAULT_ERROR;
    }

    return FAULT_ERROR;
}

/* generate_fault for a chosen neuron instead of a random one: the same
   living-list swap get_victim_idx does, then the same clamp. Local addition
   (not upstream SPIRES). Failing a neuron that has already failed is an error,
   so a caller can't silently double-count. */
int fault_neuron(struct reservoir *r, struct fault_system *fs, size_t neuron_idx)
{
    if (!r || !fs || neuron_idx >= r->num_neurons || r->failed_neurons[neuron_idx])
        return FAULT_ERROR;
    if (fs->fault_type == NONE)
        return FAULT_OK;
    size_t pos = fs->num_living;
    for (size_t i = 0; i < fs->num_living; i++) {
        if (fs->living_neurons[i] == neuron_idx) {
            pos = i;
            break;
        }
    }
    if (pos == fs->num_living) {
        fprintf(stderr,
            "Error failing neuron %zu: healthy but absent from the living half of living_neurons.\n",
            neuron_idx);
        return FAULT_ERROR;
    }
    fs->living_neurons[pos] = fs->living_neurons[fs->num_living-1];
    fs->living_neurons[fs->num_living-1] = neuron_idx;
    fs->num_living -= 1;
    switch(fs->fault_type) {
        case CLAMP_LOW:
            r->failed_neurons[neuron_idx] = 1;
            return process_clamp_low(r, fs, neuron_idx);
        default:
            fprintf(stderr, "Error: unknown fault type: %d\n", (int)fs->fault_type);
            return FAULT_ERROR;
    }
}

/* Picks a random neuron, sets its edge weights to 0 and its state to V_reset/V_0 */
int process_clamp_low(struct reservoir *r, struct fault_system *fs, size_t neuron_idx)
{
    if (getenv("SPIRES_FAULTLOG")) fprintf(stderr, "FAULTLOG neuron=%zu\n", neuron_idx);
    // clamping logic is dependent on neuron type
    // neuron->V = ...; would be way easier with templates..
    switch(r->neuron_type) {
        case LIF_DISCRETE: {
            struct lif_discrete_neuron *neuron = (struct lif_discrete_neuron *)(r->neurons[neuron_idx]);
            // save neuron state
            fs->saved_neuron_states[neuron_idx] = (struct neuron_state){neuron->V, neuron->spike};
            neuron->V = fs->neuron_fault_state;
            break;
        }
        case LIF_BIO: {
            struct lif_bio_neuron *neuron = (struct lif_bio_neuron *)(r->neurons[neuron_idx]);
            // save neuron state
            fs->saved_neuron_states[neuron_idx] = (struct neuron_state){neuron->V, neuron->spike};
            neuron->V = fs->neuron_fault_state;
            break;
        }
        case FLIF_CAPUTO: {
            struct flif_caputo_neuron *neuron = (struct flif_caputo_neuron *)(r->neurons[neuron_idx]);
            // save neuron state
            fs->saved_neuron_states[neuron_idx] = (struct neuron_state){neuron->V, neuron->spike};
            neuron->V = fs->neuron_fault_state;
            break;
        }
        case FLIF_GL: {
            struct flif_gl_neuron *neuron = (struct flif_gl_neuron *)(r->neurons[neuron_idx]);
            // save neuron state
            fs->saved_neuron_states[neuron_idx] = (struct neuron_state){neuron->V, neuron->spike};
            neuron->V = fs->neuron_fault_state;
            break;
        }
        case FLIF_DIFFUSIVE: {
            struct flif_diffusive_neuron *neuron = (struct flif_diffusive_neuron *)(r->neurons[neuron_idx]);
            // save neuron state
            fs->saved_neuron_states[neuron_idx] = (struct neuron_state){neuron->V, neuron->spike};
            neuron->V = fs->neuron_fault_state;
            break;
        }
        default: {
            fprintf(stderr, "Error, unknown neuron type: %d.\n", (int)r->neuron_type);
            return FAULT_ERROR;
        }
    }
    // zero all incoming and outgoing edge weights
    synapse_zero_neuron(&(r->W), neuron_idx);
    /* A failed neuron never spikes again, but its STDP traces still hold what
       it accumulated before failing, and the plasticity pass visits every
       stored weight, the zeroed ones included: a leftover x_pre_e would
       potentiate its severed outgoing weights back up. With its traces
       zeroed, every STDP term touching it is exactly zero, so they stay
       severed. Restored neurons start with zero traces. */
    if (r->plasticity.type != PLASTICITY_NONE && r->plasticity.x_pre_e) {
        r->plasticity.x_pre_e[neuron_idx]  = 0.0;
        r->plasticity.x_post_e[neuron_idx] = 0.0;
        r->plasticity.x_pre_i[neuron_idx]  = 0.0;
        r->plasticity.x_post_i[neuron_idx] = 0.0;
    }
    return FAULT_OK;
}

void reset_fault_system(struct reservoir *r, struct fault_system *fs)
{
    fs->num_living = r->num_neurons;
    for (size_t i = 0; i < r->num_neurons; i++) {
        fs->living_neurons[i] = i;
        fs->saved_neuron_states[i] = (struct neuron_state){0, 0};
    }
}

/* Inverse of the clamping switch in process_clamp_low. Not expressible through
   set_neuron_state() alone, which only reaches V and not spike. */
static void restore_neuron_state(struct reservoir *r, size_t neuron_idx, struct neuron_state s)
{
    switch(r->neuron_type) {
        case LIF_DISCRETE: {
            struct lif_discrete_neuron *neuron = (struct lif_discrete_neuron *)(r->neurons[neuron_idx]);
            neuron->V = s.V;
            neuron->spike = s.spike;
            break;
        }
        case LIF_BIO: {
            struct lif_bio_neuron *neuron = (struct lif_bio_neuron *)(r->neurons[neuron_idx]);
            neuron->V = s.V;
            neuron->spike = s.spike;
            break;
        }
        case FLIF_CAPUTO: {
            struct flif_caputo_neuron *neuron = (struct flif_caputo_neuron *)(r->neurons[neuron_idx]);
            neuron->V = s.V;
            neuron->spike = s.spike;
            break;
        }
        case FLIF_GL: {
            struct flif_gl_neuron *neuron = (struct flif_gl_neuron *)(r->neurons[neuron_idx]);
            neuron->V = s.V;
            neuron->spike = s.spike;
            break;
        }
        case FLIF_DIFFUSIVE: {
            struct flif_diffusive_neuron *neuron = (struct flif_diffusive_neuron *)(r->neurons[neuron_idx]);
            neuron->V = s.V;
            neuron->spike = s.spike;
            break;
        }
    }
}

/* Undoes a CLAMP_LOW fault on a single neuron. Restoring a healthy neuron is a
   no-op, not an error, so callers can loop over indices without pre-filtering. */
int restore_neuron(struct reservoir *r, struct fault_system *fs, size_t neuron_idx)
{
    if (!r || !fs || !fs->W_restore || neuron_idx >= r->num_neurons) {
        fprintf(stderr, "Error restoring neuron: bad arguments.\n");
        return FAULT_ERROR;
    }
    if (!r->failed_neurons[neuron_idx])
        return FAULT_OK;

    size_t n = r->num_neurons;

    /* get_victim_idx parks victims in living_neurons[num_living, n), so the
       neuron must be findable there. Locate it before mutating anything, so a
       desynced fault system fails without half-restoring the reservoir. */
    size_t pos = n;
    for (size_t i = fs->num_living; i < n; i++) {
        if (fs->living_neurons[i] == neuron_idx) {
            pos = i;
            break;
        }
    }
    if (pos == n) {
        fprintf(stderr,
            "Error restoring neuron %zu: marked failed but absent from the dead half of living_neurons.\n",
            neuron_idx);
        return FAULT_ERROR;
    }

    double *row = malloc(n * sizeof(double));
    double *col = malloc(n * sizeof(double));
    if (!row || !col) {
        fprintf(stderr, "Error allocating scratch memory to restore neuron %zu.\n", neuron_idx);
        free(row);
        free(col);
        return FAULT_ERROR;
    }
    /* Edges shared with a neuron that is still failed have to stay severed --
       otherwise restoring A would silently revive the A<->B edges while B is
       still dead. Everything else comes back at its pristine value. */
    for (size_t j = 0; j < n; j++) {
        int still_dead = r->failed_neurons[j] && j != neuron_idx;
        row[j] = still_dead ? 0.0 : fs->W_restore[neuron_idx * n + j];
        col[j] = still_dead ? 0.0 : fs->W_restore[j * n + neuron_idx];
    }
    synapse_restore_neuron(&(r->W), neuron_idx, row, col);
    free(row);
    free(col);

    restore_neuron_state(r, neuron_idx, fs->saved_neuron_states[neuron_idx]);
    r->failed_neurons[neuron_idx] = 0;

    /* Swap the neuron back into the living half. */
    fs->living_neurons[pos] = fs->living_neurons[fs->num_living];
    fs->living_neurons[fs->num_living] = neuron_idx;
    fs->num_living += 1;

    return FAULT_OK;
}

/* Undoes every outstanding fault. Restoring one neuron at a time is safe in any
   order: each restore masks against whoever is still failed, so by the last one
   every edge is back at its pristine value. */
int restore_reservoir(struct reservoir *r, struct fault_system *fs)
{
    if (!r || !fs)
        return FAULT_ERROR;

    int status = FAULT_OK;
    for (size_t i = 0; i < r->num_neurons; i++) {
        if (r->failed_neurons[i] && restore_neuron(r, fs, i) != FAULT_OK)
            status = FAULT_ERROR;
    }
    return status;
}
