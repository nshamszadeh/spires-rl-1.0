#ifndef FAULT_H
#define FAULT_H

#include <stdlib.h>
#include "reservoir.h"

// Various ways neurons/edges in the reservoir can fail.
enum fault_type {
    NONE,
    // TRANSIENT, // Brief change in randomly chosen neuron structure/behavior (potential application: simulating radioactive environment)
    // METASTABLE, // Fast, (possibly) random spiking of randomly chosen neuron.
    // EDGE_FAULT, // Random neuron's individual edge weight(s) set to 0
    // GLOBAL, // Gaussian pertubation vector multiplied to all nonzero synapses
    // CLAMP_HIGH, // Clamps a random neuron's weights and activation to V_th
    CLAMP_LOW // Clamps a random neuron's weights to 0 and activation to V_rest (or 0, not sure yet...)
};

/* Return codes for generate_fault/process_clamp_low. EXHAUSTED is distinct from
   OK so a caller looping "inject until dead" can tell it has run out of victims
   rather than spinning forever on a silent no-op. */
enum fault_result {
    FAULT_OK = 0,        // a fault was injected (or the fault type is NONE)
    FAULT_ERROR = 1,     // something went wrong
    FAULT_EXHAUSTED = 2  // every neuron has already failed; nothing was injected
};

struct neuron_state {
    double V;
    double spike;
};


/* Internal fault structure. 
   fault_type: Specifies what kind of fault this system simulates 
   failed_neuron: failed_neurons[i] == 1 => reservoir->neurons[i] has failed
   saved_neuron_states: Points to neurons pre-fault state (V, spike), used to restore functional neurons.
 */
 struct fault_system {
    enum fault_type fault_type;
    unsigned int neuron_rng_state; // seed for randomly picking a functioning neuron to inject fault
    size_t *living_neurons; // keeps track of indices of non-faulted neurons
    struct neuron_state *saved_neuron_states; // Before initiating a fault on a neuron, save neuron state here to restore later
    /* Dense num_neurons^2 snapshot of the weight matrix, taken at
       create_fault_system time and never written again -- restore_neuron reads
       the pristine values back out of it. Taken as a snapshot rather than saving
       each victim's row/column at fault time so restores are order-independent:
       a per-victim save would already contain the zeros left by earlier faults.
       NOTE: this snapshots the weights as they are when the fault system is
       created, so create it only after the reservoir is fully initialized, and
       recreate it if you rescale the weights afterwards. */
    double *W_restore;
    size_t num_living; // number of non-faulted neurons
    double neuron_fault_state; // IMPORTANT: as of now, only works if neuron_fault_state == 0.0
};

struct fault_system *create_fault_system(struct reservoir *r, enum fault_type fault_type, unsigned int rng_state, double neuron_fault_state);
void free_fault_system(struct fault_system *fs);
void reset_fault_system(struct reservoir *r, struct fault_system *fs);
int process_clamp_low(struct reservoir *r, struct fault_system *fs, size_t neuron_idx);
size_t get_victim_idx(struct fault_system *fs);
int generate_fault(struct reservoir *r, struct fault_system *fs);
/* generate_fault for a chosen neuron (local addition). */
int fault_neuron(struct reservoir *r, struct fault_system *fs, size_t neuron_idx);
/* Undo a fault. restore_neuron un-fails one neuron; restore_reservoir un-fails
   every currently-failed neuron. Both put back the synaptic weights, the
   membrane state saved at fault time, and the living-neuron bookkeeping. */
int restore_neuron(struct reservoir *r, struct fault_system *fs, size_t neuron_idx);
int restore_reservoir(struct reservoir *r, struct fault_system *fs);

#endif // FAULT_H