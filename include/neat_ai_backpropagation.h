/*
 * neat_ai_backpropagation — C ABI for an in-process `trainDir` (issue #84).
 *
 * Built as `libneat_ai_backpropagation.{dylib,so,dll}` by
 * `cargo build --release -p neat_ai_backpropagation`. The Rust side lives in
 * `backpropagation/src/ffi.rs`; keep the two in step.
 *
 * Contract
 * --------
 * Requests and responses are UTF-8 JSON. The library allocates every buffer it
 * hands back and the caller returns it through `neat_backprop_buffer_free` —
 * never `free()` it directly, the allocators need not match. Buffers are not
 * NUL-terminated: read exactly `len` bytes.
 *
 * Every call is fail-loud. On a non-zero status the out buffer holds a UTF-8
 * error message instead of a response, so a failure is never an empty buffer
 * the caller has to interpret.
 *
 * Copyright the NEAT-AI-Backpropagation authors. Licensed under Apache-2.0.
 */

#ifndef NEAT_AI_BACKPROPAGATION_H
#define NEAT_AI_BACKPROPAGATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes returned by neat_backprop_train. */
#define NEAT_BACKPROP_OK 0
#define NEAT_BACKPROP_ERR_INVALID_ARGUMENT 1
#define NEAT_BACKPROP_ERR_TRAIN_FAILED 2
#define NEAT_BACKPROP_ERR_PANIC 3

/*
 * An owned byte buffer produced by this library.
 *
 * `data` is NULL only before a call populates it. `capacity` exists so the
 * library can free `data`; callers should read `len` bytes and ignore it.
 */
typedef struct NeatBackpropBuffer {
  uint8_t *data;
  size_t len;
  size_t capacity;
} NeatBackpropBuffer;

/* Revision of the JSON wire contract this library implements. */
uint32_t neat_backprop_abi_version(void);

/*
 * Crate version as a static NUL-terminated string. Valid for the life of the
 * loaded library; do not free it.
 */
const char *neat_backprop_version(void);

/*
 * Run one `trainDir` epoch loop.
 *
 * `request` points at `request_len` bytes of UTF-8 JSON:
 *
 *   {
 *     "creatureJson": "<UUID-only creature JSON>",   // required
 *     "trainingData": "/path/to/bin/dir",            // required
 *     "outputDir": "/path/for/best.json+journal",    // required
 *     "epochs": 1,
 *     "maxRecords": null,
 *     "seed": 1,
 *     "disableRandomSamples": false,
 *     "learningRate": 0.01,
 *     "learningRateStrategy": "fixed",  // decay | adaptive | warmRestart
 *     "learningRateDecay": 0.95,
 *     "normaliseGradients": false,
 *     "maximumBiasAdjustmentScale": 1.0,
 *     "maximumWeightAdjustmentScale": 1.0,
 *     "stepScale": 0.01,
 *     // Whole-creature update budget (#109). Every field is optional and off
 *     // by default, which is the historical fixed-step apply. A configured
 *     // budget rescales the epoch's whole proposal to fit it.
 *     "trustRegion": {
 *       "l2": null,              // max L2 norm of the update
 *       "rms": null,             // max RMS per-gene delta
 *       "relativeRms": null,     // max RMS relative change (delta / value)
 *       "biasL2": null,          // max L2 norm of the bias genes
 *       "weightL2": null,        // max L2 norm of the weight genes
 *       "maxChangedGenes": null  // max genes one update may move
 *     },
 *     "stepScaleLadder": [],  // scorer-guided step-scale grid; [] = line search
 *     "outputsOnly": false,
 *     "hiddenOnly": false,
 *     "acceptance": "mse",          // mse | scorer ("scorer" needs "scorer")
 *     "minScoreImprovement": 1e-6,  // scorer-guided accept epsilon
 *     "msePreScreen": false,        // skip scoring a candidate MSE rejected
 *     "acceptAlways": false,        // refused together with "scorer"
 *     "maxBacktracks": 6,
 *     "scorer": null,        // optional rust_scorer binary
 *     "traceStore": null     // optional NEAT-AI traceStore directory
 *   }
 *
 * Omitted fields take the CLI `train` defaults shown above. An unknown field is
 * rejected rather than ignored.
 *
 * On NEAT_BACKPROP_OK, `*out` holds the response JSON:
 *
 *   {
 *     "abiVersion": 1,
 *     "version": "0.1.17",
 *     "bestCreatureJson": "<UUID-only creature JSON>",
 *     "baselineMse": 0.0,
 *     "bestMse": 0.0,
 *     "acceptedEpochs": 0,
 *     "epochs": 1,
 *     "bestPath": "<outputDir>/best.json",
 *     "journalPath": "<outputDir>/journal.jsonl",
 *     "bestTracePath": null,
 *     "failedTraceDir": null,
 *     "baselineScore": null,
 *     "bestScore": null
 *   }
 *
 * On any other status, `*out` holds the error message.
 *
 * `out` must not be NULL (a NULL `out` returns
 * NEAT_BACKPROP_ERR_INVALID_ARGUMENT with no message, since there is nowhere to
 * write one) and must not already own a buffer.
 */
int32_t neat_backprop_train(const uint8_t *request, size_t request_len,
                            NeatBackpropBuffer *out);

/*
 * Release a buffer produced by this library and reset it to empty. Safe to call
 * with NULL or on an already-freed buffer.
 */
void neat_backprop_buffer_free(NeatBackpropBuffer *buffer);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NEAT_AI_BACKPROPAGATION_H */
