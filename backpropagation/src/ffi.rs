//! C ABI for an in-process `trainDir` (issue #84).
//!
//! NEAT-AI reached this crate by spawning the CLI. That costs a process launch
//! and a JSON temp directory on every memetic `trainDir`, and it cannot carry
//! richer in-process control. This module exposes the same `train` contract as
//! a `cdylib` symbol set a Deno FFI (or any C) caller can `dlopen`.
//!
//! The wire format is UTF-8 JSON in and UTF-8 JSON out, carried in owned
//! buffers with explicit lengths ([`NeatBackpropBuffer`]) so the caller never
//! has to guess a length or match an allocator. Every call is fail-loud: a bad
//! request, an unreadable corpus, or a panic inside the trainer all return a
//! non-zero status **and** an error message in the same out buffer — there is
//! no silent fallback and no partial success.
//!
//! ```text
//! neat_backprop_abi_version() -> u32          ABI revision (this file)
//! neat_backprop_version()     -> *const char  crate version, NUL-terminated
//! neat_backprop_train(req, req_len, out) -> i32
//! neat_backprop_buffer_free(out)
//! neat_backprop_capabilities() -> *const char  capability JSON, NUL-terminated
//! ```
//!
//! `neat_backprop_capabilities` is absent from older trainers, which have no
//! deadline support (#214). A caller probes for it before sending
//! `deadlineEpochMs`.
//!
//! See [`include/neat_ai_backpropagation.h`](../../../include/neat_ai_backpropagation.h)
//! for the C declarations.

use crate::backprop::{ApplyOptions, BackpropConfig, LearningRateStrategy};
use crate::deadline::{Deadline, TimeoutPhase};
use crate::mse::ErrorMeasure;
use crate::scorer::ScoreResult;
use crate::train::{
    BEST_TRACE_FILE, DEFAULT_MIN_SCORE_IMPROVEMENT, DEFAULT_STEP_SCALE, FAILED_TRACE_DIR,
    TrainCreature, TrainRequest, TrainResult, TrainStatus, resolve_acceptance, run_train_until,
};
use crate::trust_region::TrustRegion;
use serde::de::{Deserializer, Error as _};
use serde::{Deserialize, Serialize};
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::OnceLock;

/// Revision of the wire contract in this module.
///
/// Bumped whenever a released request/response field changes meaning or is
/// removed, so a caller can refuse a library it was not built against.
pub const NEAT_BACKPROP_ABI_VERSION: u32 = 1;

/// The call succeeded; the out buffer holds the response JSON.
pub const NEAT_BACKPROP_OK: i32 = 0;

/// The request pointer, its length, its encoding, or its JSON was unusable.
pub const NEAT_BACKPROP_ERR_INVALID_ARGUMENT: i32 = 1;

/// The trainer itself failed; the out buffer holds its error message.
pub const NEAT_BACKPROP_ERR_TRAIN_FAILED: i32 = 2;

/// A panic was caught at the boundary rather than unwinding into the caller.
pub const NEAT_BACKPROP_ERR_PANIC: i32 = 3;

/// Crate version with the terminating NUL [`neat_backprop_version`] hands out.
const VERSION_WITH_NUL: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

/// An owned byte buffer handed to the caller.
///
/// The library allocates it and the caller must return it through
/// [`neat_backprop_buffer_free`] — never `free()` it directly, since Rust's
/// allocator need not be the C one. `data` is never NUL-terminated: read
/// exactly `len` bytes.
#[repr(C)]
#[derive(Debug)]
pub struct NeatBackpropBuffer {
    /// Pointer to `len` UTF-8 bytes, or null when the buffer is empty.
    pub data: *mut u8,
    /// Number of readable bytes at `data`.
    pub len: usize,
    /// Allocated capacity — needed to free `data`; callers should ignore it.
    pub capacity: usize,
}

impl NeatBackpropBuffer {
    /// An empty buffer that owns nothing.
    fn empty() -> Self {
        Self {
            data: ptr::null_mut(),
            len: 0,
            capacity: 0,
        }
    }

    /// Take ownership of `bytes` and expose it to the caller.
    fn from_vec(bytes: Vec<u8>) -> Self {
        let mut bytes = bytes;
        let buffer = Self {
            data: bytes.as_mut_ptr(),
            len: bytes.len(),
            capacity: bytes.capacity(),
        };
        std::mem::forget(bytes);
        buffer
    }
}

/// Learning-rate schedule selectable over the ABI (mirrors the CLI flag).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AbiLearningRateStrategy {
    /// Constant learning rate.
    #[default]
    Fixed,
    /// Multiplicative decay each epoch.
    Decay,
    /// Boost / shrink from epoch-to-epoch MSE feedback.
    Adaptive,
    /// Decay with a periodic warm restart.
    WarmRestart,
}

impl AbiLearningRateStrategy {
    /// Map the wire value onto the library strategy.
    fn to_config(self) -> LearningRateStrategy {
        match self {
            Self::Fixed => LearningRateStrategy::Fixed,
            Self::Decay => LearningRateStrategy::Decay,
            Self::Adaptive => LearningRateStrategy::Adaptive,
            Self::WarmRestart => LearningRateStrategy::WarmRestart,
        }
    }
}

/// Default epoch count (CLI `--epochs`).
fn default_epochs() -> u64 {
    1
}

/// Default seed (CLI `--seed`).
fn default_seed() -> u64 {
    1
}

/// Default learning rate (CLI `--learning-rate`).
fn default_learning_rate() -> f64 {
    0.01
}

/// Default per-epoch decay factor (CLI `--learning-rate-decay`).
fn default_learning_rate_decay() -> f64 {
    0.95
}

/// Default bias / weight adjustment clamp (CLI `--maximum-*-adjustment-scale`).
fn default_adjustment_scale() -> f64 {
    1.0
}

/// Default apply step scale (CLI `--step-scale`).
fn default_step_scale() -> f64 {
    DEFAULT_STEP_SCALE
}

/// Default backtracking budget (CLI `--max-backtracks`).
fn default_max_backtracks() -> u32 {
    6
}

/// Default minimum scorer gain (CLI `--min-score-improvement`).
fn default_min_score_improvement() -> f64 {
    DEFAULT_MIN_SCORE_IMPROVEMENT
}

/// Acceptance mode selectable over the ABI (mirrors the CLI flag, #104).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AbiAcceptanceMode {
    /// Training-slice MSE decides accept / rollback.
    #[default]
    Mse,
    /// `NEAT-AI-scorer` fitness decides accept / rollback.
    Scorer,
}

/// JSON request for [`train_from_json`] — one `trainDir` run.
///
/// Field names are camelCase on the wire and every optional field defaults to
/// the matching CLI `train` flag, so a NEAT-AI caller forwarding the flags it
/// already forwards gets identical behaviour. Unknown fields are rejected: a
/// typo must fail loudly rather than silently train with a default.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrainAbiRequest {
    /// UUID-only creature JSON text (not a path).
    pub creature_json: String,
    /// Directory of little-endian f32 `.bin` records.
    pub training_data: PathBuf,
    /// Output directory for `best.json` and `journal.jsonl`.
    pub output_dir: PathBuf,
    /// Epochs to run.
    #[serde(default = "default_epochs")]
    pub epochs: u64,
    /// Record cap per epoch, honoured as a rate over the whole corpus.
    #[serde(default)]
    pub max_records: Option<u64>,
    /// Sparse-selection and record-sampling seed.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Take each file's leading records instead of a seeded random draw.
    #[serde(default)]
    pub disable_random_samples: bool,
    /// Initial learning rate.
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    /// Learning-rate schedule across epochs.
    #[serde(default)]
    pub learning_rate_strategy: AbiLearningRateStrategy,
    /// Per-epoch decay factor for the decay / warm-restart strategies.
    #[serde(default = "default_learning_rate_decay")]
    pub learning_rate_decay: f64,
    /// Divide multi-path gradients by `sqrt(path count)`.
    #[serde(default)]
    pub normalise_gradients: bool,
    /// Maximum |Δbias| per apply.
    #[serde(default = "default_adjustment_scale")]
    pub maximum_bias_adjustment_scale: f64,
    /// Maximum |Δweight| per apply.
    #[serde(default = "default_adjustment_scale")]
    pub maximum_weight_adjustment_scale: f64,
    /// Multiply `(proposed − current)` by this factor before writing.
    #[serde(default = "default_step_scale")]
    pub step_scale: f64,
    /// Whole-creature update budget the apply is rescaled to fit (#109).
    ///
    /// Every budget is off by default, which is the fixed-step parity mode —
    /// the candidate is exactly what `stepScale` alone produces.
    #[serde(default)]
    pub trust_region: TrustRegion,
    /// Step scales to score as a ladder under `acceptance: "scorer"` (#106).
    ///
    /// Empty (the default) keeps the backtracking line search; a non-empty
    /// grid supersedes it and is refused on an `acceptance: "mse"` run.
    #[serde(default)]
    pub step_scale_ladder: Vec<f64>,
    /// Apply only output neurons and synapses that target them.
    #[serde(default)]
    pub outputs_only: bool,
    /// Apply only hidden / constant genes (skip output).
    #[serde(default)]
    pub hidden_only: bool,
    /// What decides accept / rollback — slice MSE, or the scorer (#104).
    #[serde(default)]
    pub acceptance: AbiAcceptanceMode,
    /// Minimum scorer gain to keep a candidate under `acceptance: "scorer"`.
    #[serde(default = "default_min_score_improvement")]
    pub min_score_improvement: f64,
    /// Drop a candidate whose slice MSE did not fall before scoring it.
    #[serde(default)]
    pub mse_pre_screen: bool,
    /// Keep the applied creature even if slice MSE rose.
    ///
    /// Refused together with `acceptance: "scorer"`.
    #[serde(default)]
    pub accept_always: bool,
    /// Halvings of step scale to retry a rejected apply with.
    #[serde(default = "default_max_backtracks")]
    pub max_backtracks: u32,
    /// Optional `rust_scorer` binary for a before/after score.
    #[serde(default)]
    pub scorer: Option<PathBuf>,
    /// Optional NEAT-AI `traceStore` directory for `CreatureTrace` artefacts.
    #[serde(default)]
    pub trace_store: Option<PathBuf>,
    /// Absolute deadline in milliseconds since the Unix epoch (#214).
    ///
    /// Same unit and expiry as [`Deadline::at_epoch_ms`]. Absent or `null` runs
    /// unbounded. Once the deadline passes the run stops, keeps the learning it
    /// has accepted, and reports `timedOut: true` with a `timeout` record.
    #[serde(default, deserialize_with = "deserialize_deadline_epoch_ms")]
    pub deadline_epoch_ms: Option<u64>,
}

impl Default for TrainAbiRequest {
    fn default() -> Self {
        Self {
            creature_json: String::new(),
            training_data: PathBuf::new(),
            output_dir: PathBuf::new(),
            epochs: default_epochs(),
            max_records: None,
            seed: default_seed(),
            disable_random_samples: false,
            learning_rate: default_learning_rate(),
            learning_rate_strategy: AbiLearningRateStrategy::default(),
            learning_rate_decay: default_learning_rate_decay(),
            normalise_gradients: false,
            maximum_bias_adjustment_scale: default_adjustment_scale(),
            maximum_weight_adjustment_scale: default_adjustment_scale(),
            step_scale: default_step_scale(),
            trust_region: TrustRegion::default(),
            step_scale_ladder: Vec::new(),
            outputs_only: false,
            hidden_only: false,
            acceptance: AbiAcceptanceMode::default(),
            min_score_improvement: default_min_score_improvement(),
            mse_pre_screen: false,
            accept_always: false,
            max_backtracks: default_max_backtracks(),
            scorer: None,
            trace_store: None,
            deadline_epoch_ms: None,
        }
    }
}

/// Deserialise `deadlineEpochMs`: absent or `null` is unbounded, otherwise it
/// must be a non-negative integer number of milliseconds (#214).
///
/// `serde_json::Value::as_u64` accepts only a non-negative integer, so a
/// negative, fractional, string, boolean or object value is refused here rather
/// than silently training unbounded.
fn deserialize_deadline_epoch_ms<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    match Option::<serde_json::Value>::deserialize(deserializer)? {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => value.as_u64().map(Some).ok_or_else(|| {
            D::Error::custom(format!(
                "deadlineEpochMs must be a non-negative integer number of milliseconds since the Unix epoch, got {value}"
            ))
        }),
    }
}

/// How an error figure was measured, as reported on the wire (#214).
///
/// Only `Full` is a whole-corpus score. `Partial` covers the records scored
/// before a deadline, and `Unmeasured` means nothing was scored at all.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum MeasureReport {
    /// No record was scored before the deadline.
    Unmeasured,
    /// Scored over only the first `records` records before the deadline.
    Partial {
        /// Mean squared error over the `records` scored records.
        mse: f64,
        /// Records scored before the deadline expired.
        records: u64,
    },
    /// Scored over every selected record.
    Full {
        /// Mean squared error over every selected record.
        mse: f64,
        /// Records in the selection.
        records: u64,
    },
}

impl From<ErrorMeasure> for MeasureReport {
    fn from(measure: ErrorMeasure) -> Self {
        match measure {
            ErrorMeasure::Unmeasured => Self::Unmeasured,
            ErrorMeasure::Partial { mse, records } => Self::Partial { mse, records },
            ErrorMeasure::Full { mse, records } => Self::Full { mse, records },
        }
    }
}

/// Where and how a deadline cut a run short (#214).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeoutReport {
    /// The trainer step at which expiry was observed.
    pub phase: TimeoutPhase,
    /// Journalled non-partial epochs completed before the deadline.
    pub completed_epochs: u64,
    /// Whether at least one epoch, full or partial, was accepted.
    pub weights_changed: bool,
    /// The absolute deadline the run was given, in milliseconds since the Unix epoch.
    pub deadline_epoch_ms: u64,
}

/// JSON response from a successful [`train_from_json`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainAbiResponse {
    /// ABI revision that produced this response.
    pub abi_version: u32,
    /// Crate version that produced this response.
    pub version: String,
    /// Best creature JSON — the exact bytes written to `best.json`.
    pub best_creature_json: String,
    /// Baseline MSE before any apply. Present only when it was measured over the
    /// whole corpus (`baselineMeasure.kind == "full"`); `null` otherwise.
    pub baseline_mse: Option<f64>,
    /// Best MSE after the run. Present only when it was measured over the whole
    /// corpus (`bestMeasure.kind == "full"`); `null` otherwise.
    pub best_mse: Option<f64>,
    /// How the baseline error was measured: full, partial or unmeasured.
    pub baseline_measure: MeasureReport,
    /// How the best error was measured, with the same states as `baselineMeasure`.
    pub best_measure: MeasureReport,
    /// Epochs that accepted an apply.
    pub accepted_epochs: u64,
    /// Epochs requested.
    pub epochs: u64,
    /// Path of the written `best.json`.
    pub best_path: PathBuf,
    /// Path of the written `journal.jsonl`.
    pub journal_path: PathBuf,
    /// Best-epoch `CreatureTrace`, when a trace store was requested and written.
    pub best_trace_path: Option<PathBuf>,
    /// Failed-candidate trace directory, when it was written.
    pub failed_trace_dir: Option<PathBuf>,
    /// Baseline scorer result, when a scorer was supplied.
    pub baseline_score: Option<ScoreResult>,
    /// Best-creature scorer result, when a scorer was supplied.
    pub best_score: Option<ScoreResult>,
    /// Whether the deadline stopped the run before every epoch completed.
    pub timed_out: bool,
    /// Where and how the deadline cut the run short; `null` when it completed.
    pub timeout: Option<TimeoutReport>,
    /// Journalled non-partial epochs, including rejected epochs and one whose
    /// backtracking search was cut short by a deadline.
    pub completed_epochs: u64,
}

/// Run one `trainDir` epoch loop described by a JSON request.
///
/// This is the ABI's whole behaviour, callable from Rust so it can be tested
/// without going through raw pointers. Errors are returned as messages, never
/// swallowed.
pub fn train_from_json(request: &str) -> Result<String, String> {
    let request: TrainAbiRequest =
        serde_json::from_str(request).map_err(|e| format!("invalid train request JSON: {e}"))?;
    let response = train(&request)?;
    serde_json::to_string(&response).map_err(|e| format!("failed to encode train response: {e}"))
}

/// Run one `trainDir` epoch loop from a decoded request.
///
/// A `deadlineEpochMs` bounds the run. A run that runs out of time still returns
/// `Ok`, with `timedOut` set on the response (#214).
pub fn train(request: &TrainAbiRequest) -> Result<TrainAbiResponse, String> {
    // Refuses a scorer knob on an MSE run rather than ignoring it (#104).
    let acceptance = resolve_acceptance(
        request.acceptance == AbiAcceptanceMode::Scorer,
        request.min_score_improvement,
        request.mse_pre_screen,
    )?;
    let config = BackpropConfig {
        learning_rate: request.learning_rate,
        initial_learning_rate: request.learning_rate,
        learning_rate_strategy: request.learning_rate_strategy.to_config(),
        learning_rate_decay: request.learning_rate_decay,
        maximum_bias_adjustment_scale: request.maximum_bias_adjustment_scale,
        maximum_weight_adjustment_scale: request.maximum_weight_adjustment_scale,
        normalise_gradients: request.normalise_gradients,
        ..BackpropConfig::default()
    };
    let deadline = request.deadline_epoch_ms.map(Deadline::at_epoch_ms);
    let result = run_train_until(
        TrainRequest {
            creature: TrainCreature::Json(&request.creature_json),
            training_data: &request.training_data,
            config: &config,
            epochs: request.epochs,
            max_records: request.max_records,
            seed: request.seed,
            disable_random_samples: request.disable_random_samples,
            output_dir: &request.output_dir,
            scorer: request.scorer.as_deref(),
            apply: ApplyOptions {
                step_scale: request.step_scale,
                outputs_only: request.outputs_only,
                hidden_only: request.hidden_only,
            },
            trust_region: request.trust_region,
            acceptance,
            accept_always: request.accept_always,
            max_backtracks: request.max_backtracks,
            step_scale_ladder: &request.step_scale_ladder,
            trace_store: request.trace_store.as_deref(),
        },
        deadline.as_ref(),
    )?;

    Ok(build_train_response(
        result,
        request.epochs,
        &request.output_dir,
        request.trace_store.as_deref(),
        request.deadline_epoch_ms,
    ))
}

/// Build the ABI response from a finished run (#214).
///
/// Infallible, since `result` is already a complete outcome. `baselineMse` and
/// `bestMse` are `Some` only for a `Full` measure, so a partial figure is never
/// labelled a full-corpus score. `deadline_epoch_ms` is the deadline the run was
/// given, if any, and is echoed in `timeout` when the run timed out.
pub fn build_train_response(
    result: TrainResult,
    epochs: u64,
    output_dir: &Path,
    trace_store: Option<&Path>,
    deadline_epoch_ms: Option<u64>,
) -> TrainAbiResponse {
    // Only a run that asked for a trace store can claim these artefacts —
    // otherwise a stale file left in the output directory would be reported as
    // this run's trace.
    let best_trace_path = trace_store.and_then(|_| {
        let path = output_dir.join(BEST_TRACE_FILE);
        path.is_file().then_some(path)
    });
    let failed_trace_dir = trace_store.map(|store| store.join(FAILED_TRACE_DIR));
    let (timed_out, timeout, completed_epochs) = match result.status {
        TrainStatus::Completed => (false, None, result.completed_epochs),
        TrainStatus::TimedOut {
            phase,
            completed_epochs,
            weights_changed,
        } => (
            true,
            Some(TimeoutReport {
                phase,
                completed_epochs,
                weights_changed,
                deadline_epoch_ms: deadline_epoch_ms
                    .expect("a run only times out under a deadline"),
            }),
            completed_epochs,
        ),
    };
    TrainAbiResponse {
        abi_version: NEAT_BACKPROP_ABI_VERSION,
        version: env!("CARGO_PKG_VERSION").to_string(),
        best_creature_json: result.best_json,
        baseline_mse: result.baseline.full_mse(),
        best_mse: result.best.full_mse(),
        baseline_measure: result.baseline.into(),
        best_measure: result.best.into(),
        accepted_epochs: result.accepted_epochs,
        epochs,
        best_path: output_dir.join("best.json"),
        journal_path: output_dir.join("journal.jsonl"),
        best_trace_path,
        failed_trace_dir: failed_trace_dir.filter(|dir| dir.is_dir()),
        baseline_score: result.baseline_score,
        best_score: result.best_score,
        timed_out,
        timeout,
        completed_epochs,
    }
}

/// ABI revision this library implements ([`NEAT_BACKPROP_ABI_VERSION`]).
#[unsafe(no_mangle)]
pub extern "C" fn neat_backprop_abi_version() -> u32 {
    NEAT_BACKPROP_ABI_VERSION
}

/// Crate version as a static NUL-terminated UTF-8 string.
///
/// The pointer is valid for the life of the loaded library and must not be
/// freed.
#[unsafe(no_mangle)]
pub extern "C" fn neat_backprop_version() -> *const c_char {
    VERSION_WITH_NUL.as_ptr().cast::<c_char>()
}

/// Revision of the deadline capability advertised by [`capabilities_json`] (#214).
pub const NEAT_BACKPROP_DEADLINE_CAPABILITY_VERSION: u32 = 1;

/// File name the CLI writes the run's [`TrainAbiResponse`] to, inside the output
/// directory, beside `best.json` (#214).
pub const TRAIN_RESULT_FILE: &str = "result.json";

/// The compact capability probe document: the ABI revision, the crate version,
/// and the optional capabilities this build supports (#214).
pub fn capabilities_json() -> String {
    serde_json::json!({
        "abiVersion": NEAT_BACKPROP_ABI_VERSION,
        "version": env!("CARGO_PKG_VERSION"),
        "capabilities": {
            "deadline": {
                "version": NEAT_BACKPROP_DEADLINE_CAPABILITY_VERSION,
                "requestField": "deadlineEpochMs",
                "cliFlag": "--deadline-epoch-ms",
                "unit": "epochMilliseconds",
                "resultFile": TRAIN_RESULT_FILE,
            }
        }
    })
    .to_string()
}

/// Capability probe: the [`capabilities_json`] document as a static
/// NUL-terminated UTF-8 string (#214).
///
/// A library that does not export this symbol is an older trainer with no
/// deadline support, so a caller must not send `deadlineEpochMs` to it. The
/// pointer is valid for the life of the loaded library and must not be freed.
#[unsafe(no_mangle)]
pub extern "C" fn neat_backprop_capabilities() -> *const c_char {
    static CAPABILITIES: OnceLock<CString> = OnceLock::new();
    CAPABILITIES
        .get_or_init(|| {
            CString::new(capabilities_json()).expect("capabilities JSON holds no NUL byte")
        })
        .as_ptr()
}

/// Run one `trainDir` epoch loop and write the JSON response into `out`.
///
/// Returns [`NEAT_BACKPROP_OK`] on success, in which case `out` holds the
/// response JSON. On any other status `out` holds a UTF-8 error message
/// instead — it is always populated when `out` is non-null, so the caller never
/// has to infer a failure from an empty buffer. Either way the caller must
/// release `out` with [`neat_backprop_buffer_free`].
///
/// # Safety
///
/// `request` must point to `request_len` initialised bytes, and `out` must
/// point to a writable, properly aligned [`NeatBackpropBuffer`] that does not
/// already own a buffer (its previous contents are overwritten, not freed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn neat_backprop_train(
    request: *const u8,
    request_len: usize,
    out: *mut NeatBackpropBuffer,
) -> i32 {
    if out.is_null() {
        // Nowhere to report to — the status code is the only channel left.
        return NEAT_BACKPROP_ERR_INVALID_ARGUMENT;
    }
    // SAFETY: `out` is non-null and the caller guarantees it is writable and
    // aligned; overwriting it is sound because it must not already own a
    // buffer.
    unsafe { out.write(NeatBackpropBuffer::empty()) };

    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if request.is_null() {
            return (
                NEAT_BACKPROP_ERR_INVALID_ARGUMENT,
                "request pointer is null".to_string(),
            );
        }
        // SAFETY: the caller guarantees `request` covers `request_len`
        // initialised bytes; the slice is only read inside this call.
        let bytes = unsafe { std::slice::from_raw_parts(request, request_len) };
        let text = match std::str::from_utf8(bytes) {
            Ok(text) => text,
            Err(e) => {
                return (
                    NEAT_BACKPROP_ERR_INVALID_ARGUMENT,
                    format!("request is not valid UTF-8: {e}"),
                );
            }
        };
        match train_from_json(text) {
            Ok(response) => (NEAT_BACKPROP_OK, response),
            Err(e) if e.starts_with("invalid train request JSON") => {
                (NEAT_BACKPROP_ERR_INVALID_ARGUMENT, e)
            }
            Err(e) => (NEAT_BACKPROP_ERR_TRAIN_FAILED, e),
        }
    }));

    let (status, payload) = match outcome {
        Ok(pair) => pair,
        Err(panic) => (NEAT_BACKPROP_ERR_PANIC, panic_message(&panic)),
    };
    // SAFETY: as above — `out` is non-null, writable and aligned, and the
    // empty buffer written earlier owns nothing.
    unsafe { out.write(NeatBackpropBuffer::from_vec(payload.into_bytes())) };
    status
}

/// Release a buffer produced by this library and reset it to empty.
///
/// Safe to call with a null pointer or on an already-freed buffer.
///
/// # Safety
///
/// `buffer` must be null or point to a writable [`NeatBackpropBuffer`] this
/// library produced and that has not already been freed by another call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn neat_backprop_buffer_free(buffer: *mut NeatBackpropBuffer) {
    if buffer.is_null() {
        return;
    }
    // SAFETY: the caller guarantees `buffer` is a writable, aligned buffer
    // this library produced, so `data`/`len`/`capacity` are the parts of a
    // `Vec<u8>` allocated by this library's allocator.
    unsafe {
        let owned = &mut *buffer;
        if !owned.data.is_null() {
            drop(Vec::from_raw_parts(owned.data, owned.len, owned.capacity));
        }
        buffer.write(NeatBackpropBuffer::empty());
    }
}

/// Best-effort text of a caught panic payload.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = panic.downcast_ref::<&str>() {
        format!("panic in neat_backprop_train: {text}")
    } else if let Some(text) = panic.downcast_ref::<String>() {
        format!("panic in neat_backprop_train: {text}")
    } else {
        "panic in neat_backprop_train".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_cli_train_flags() {
        let request: TrainAbiRequest = serde_json::from_str(
            r#"{"creatureJson":"{}","trainingData":"data","outputDir":"out"}"#,
        )
        .unwrap();
        assert_eq!(request.epochs, 1);
        assert_eq!(request.seed, 1);
        assert!(!request.disable_random_samples);
        assert!((request.learning_rate - 0.01).abs() < 1e-12);
        assert_eq!(
            request.learning_rate_strategy,
            AbiLearningRateStrategy::Fixed
        );
        assert!((request.learning_rate_decay - 0.95).abs() < 1e-12);
        assert!(!request.normalise_gradients);
        assert!((request.maximum_bias_adjustment_scale - 1.0).abs() < 1e-12);
        assert!((request.maximum_weight_adjustment_scale - 1.0).abs() < 1e-12);
        assert!((request.step_scale - DEFAULT_STEP_SCALE).abs() < 1e-12);
        assert!(request.step_scale_ladder.is_empty());
        assert!(!request.outputs_only);
        assert!(!request.hidden_only);
        assert!(!request.accept_always);
        assert_eq!(request.max_backtracks, 6);
        assert!(request.scorer.is_none());
        assert!(request.trace_store.is_none());
        // #104: MSE acceptance stays the default, so an existing caller that
        // forwards nothing new keeps the historical behaviour.
        assert_eq!(request.acceptance, AbiAcceptanceMode::Mse);
        assert!((request.min_score_improvement - DEFAULT_MIN_SCORE_IMPROVEMENT).abs() < 1e-18);
        assert!(!request.mse_pre_screen);
    }

    /// The scorer-guided flags cross the wire as camelCase and reach the
    /// trainer's own acceptance settings (#104).
    #[test]
    fn scorer_acceptance_crosses_the_wire() {
        let request: TrainAbiRequest = serde_json::from_str(
            r#"{"creatureJson":"{}","trainingData":"d","outputDir":"o",
                "acceptance":"scorer","minScoreImprovement":0.002,"msePreScreen":true}"#,
        )
        .unwrap();
        assert_eq!(request.acceptance, AbiAcceptanceMode::Scorer);
        assert!((request.min_score_improvement - 0.002).abs() < 1e-15);
        assert!(request.mse_pre_screen);
    }

    /// The step-scale ladder crosses the wire as a camelCase array and is
    /// empty — the historical line search — unless the caller asks for it
    /// (#106).
    #[test]
    fn the_step_scale_ladder_crosses_the_wire() {
        let default: TrainAbiRequest =
            serde_json::from_str(r#"{"creatureJson":"{}","trainingData":"d","outputDir":"o"}"#)
                .unwrap();
        assert!(default.step_scale_ladder.is_empty());

        let request: TrainAbiRequest = serde_json::from_str(
            r#"{"creatureJson":"{}","trainingData":"d","outputDir":"o",
                "acceptance":"scorer","stepScaleLadder":[0.0001,0.001,0.01]}"#,
        )
        .unwrap();
        assert_eq!(request.step_scale_ladder, vec![0.0001, 0.001, 0.01]);
    }

    /// A ladder on an `acceptance: "mse"` request is refused rather than
    /// silently ignored — the run would otherwise look configured (#106).
    #[test]
    fn a_ladder_on_an_mse_request_fails_loudly() {
        let err = train_from_json(
            r#"{"creatureJson":"{}","trainingData":"d","outputDir":"o",
                "stepScaleLadder":[0.001]}"#,
        )
        .unwrap_err();
        assert!(err.contains("step-scale ladder"), "unexpected error: {err}");
    }

    /// A scorer-guided request without a scorer binary is refused by the
    /// trainer, and the ABI reports that as a train failure rather than a
    /// silent MSE run (#104).
    #[test]
    fn scorer_acceptance_without_a_scorer_binary_fails_loudly() {
        let err = train_from_json(
            r#"{"creatureJson":"{}","trainingData":"d","outputDir":"o","acceptance":"scorer"}"#,
        )
        .unwrap_err();
        assert!(
            err.contains("scorer-guided acceptance"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unknown_request_fields_fail_loudly() {
        let err = train_from_json(
            r#"{"creatureJson":"{}","trainingData":"d","outputDir":"o","epocs":3}"#,
        )
        .unwrap_err();
        assert!(
            err.starts_with("invalid train request JSON"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn strategy_names_are_camel_case_on_the_wire() {
        let request: TrainAbiRequest = serde_json::from_str(
            r#"{"creatureJson":"{}","trainingData":"d","outputDir":"o","learningRateStrategy":"warmRestart"}"#,
        )
        .unwrap();
        assert_eq!(
            request.learning_rate_strategy,
            AbiLearningRateStrategy::WarmRestart
        );
        assert_eq!(
            request.learning_rate_strategy.to_config(),
            LearningRateStrategy::WarmRestart
        );
    }

    #[test]
    fn version_probe_reports_the_crate_version() {
        assert_eq!(neat_backprop_abi_version(), NEAT_BACKPROP_ABI_VERSION);
        let ptr = neat_backprop_version();
        // SAFETY: the probe hands back a static NUL-terminated string.
        let text = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(text, env!("CARGO_PKG_VERSION"));
    }

    /// A request with no `deadlineEpochMs` is unbounded, as it always was (#214).
    #[test]
    fn deadline_epoch_ms_defaults_to_unbounded() {
        let request: TrainAbiRequest =
            serde_json::from_str(r#"{"creatureJson":"{}","trainingData":"d","outputDir":"o"}"#)
                .unwrap();
        assert_eq!(request.deadline_epoch_ms, None);
        assert_eq!(TrainAbiRequest::default().deadline_epoch_ms, None);
    }

    #[test]
    fn deadline_epoch_ms_parses_an_absolute_epoch_millisecond() {
        let request: TrainAbiRequest = serde_json::from_str(
            r#"{"creatureJson":"{}","trainingData":"d","outputDir":"o","deadlineEpochMs":1700000000000}"#,
        )
        .unwrap();
        assert_eq!(request.deadline_epoch_ms, Some(1_700_000_000_000));
    }

    /// `null` is the explicit spelling of "no deadline" and must not be an error.
    #[test]
    fn a_null_deadline_is_unbounded() {
        let request: TrainAbiRequest = serde_json::from_str(
            r#"{"creatureJson":"{}","trainingData":"d","outputDir":"o","deadlineEpochMs":null}"#,
        )
        .unwrap();
        assert_eq!(request.deadline_epoch_ms, None);
    }

    /// A deadline that is not a non-negative integer must fail the call rather
    /// than train unbounded (#214).
    #[test]
    fn a_malformed_deadline_fails_loudly() {
        for value in ["-5", "1.5", "\"soon\"", "true", "{}"] {
            let text = format!(
                r#"{{"creatureJson":"{{}}","trainingData":"d","outputDir":"o","deadlineEpochMs":{value}}}"#
            );
            let err = train_from_json(&text).unwrap_err();
            assert!(
                err.starts_with("invalid train request JSON"),
                "unexpected error for {value}: {err}"
            );
            assert!(
                err.contains("deadlineEpochMs"),
                "unexpected error for {value}: {err}"
            );
        }
    }

    /// The wire report keeps the three measure states apart, and only a `Full`
    /// figure can populate the `baselineMse` / `bestMse` scalars (#214).
    #[test]
    fn measure_reports_keep_partial_and_unmeasured_apart_from_full() {
        assert_eq!(
            MeasureReport::from(ErrorMeasure::Unmeasured),
            MeasureReport::Unmeasured
        );
        assert_eq!(
            MeasureReport::from(ErrorMeasure::Partial {
                mse: 0.5,
                records: 3
            }),
            MeasureReport::Partial {
                mse: 0.5,
                records: 3
            }
        );
        assert_eq!(
            MeasureReport::from(ErrorMeasure::Full {
                mse: 0.25,
                records: 8
            }),
            MeasureReport::Full {
                mse: 0.25,
                records: 8
            }
        );

        assert_eq!(ErrorMeasure::Unmeasured.full_mse(), None);
        assert_eq!(
            ErrorMeasure::Partial {
                mse: 0.5,
                records: 3
            }
            .full_mse(),
            None,
            "a partial figure must not become a full-corpus score"
        );
        assert_eq!(
            ErrorMeasure::Full {
                mse: 0.25,
                records: 8
            }
            .full_mse(),
            Some(0.25)
        );

        let partial = serde_json::to_value(MeasureReport::from(ErrorMeasure::Partial {
            mse: 0.5,
            records: 3,
        }))
        .unwrap();
        assert_eq!(
            partial,
            serde_json::json!({"kind": "partial", "mse": 0.5, "records": 3})
        );
    }

    /// The capability probe parses, names this ABI and the deadline revision,
    /// and the static C pointer hands back the same document (#214).
    #[test]
    fn capabilities_probe_matches_the_wire_contract() {
        let caps: serde_json::Value = serde_json::from_str(&capabilities_json()).unwrap();
        assert_eq!(
            caps["abiVersion"],
            serde_json::json!(NEAT_BACKPROP_ABI_VERSION)
        );
        assert_eq!(
            caps["capabilities"]["deadline"]["version"],
            serde_json::json!(NEAT_BACKPROP_DEADLINE_CAPABILITY_VERSION)
        );
        assert_eq!(
            caps["capabilities"]["deadline"]["requestField"],
            serde_json::json!("deadlineEpochMs")
        );

        let ptr = neat_backprop_capabilities();
        // SAFETY: the probe returns a static NUL-terminated string.
        let text = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(text, capabilities_json());
    }
}
