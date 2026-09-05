//! Issue #84: the `cdylib` C ABI must round-trip a real `trainDir` run.
//!
//! These tests drive the exported symbols exactly as a Deno FFI caller does —
//! raw request bytes in, owned buffer out, buffer freed — so a regression in
//! the pointer contract fails here rather than in the consumer.

use neat_ai_backpropagation::ffi::{
    NEAT_BACKPROP_ABI_VERSION, NEAT_BACKPROP_ERR_INVALID_ARGUMENT, NEAT_BACKPROP_ERR_TRAIN_FAILED,
    NEAT_BACKPROP_OK, NeatBackpropBuffer, TrainAbiResponse, neat_backprop_abi_version,
    neat_backprop_buffer_free, neat_backprop_train, neat_backprop_version,
};
use serde_json::{Value, json};
use std::ffi::CStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// Identity chain: `input-0 → h1 → o1`, all IDENTITY squashes.
const CHAIN: &str = r#"{
  "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
  "neurons":[
    {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
    {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
  ],
  "synapses":[
    {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
    {"fromUUID":"h1","toUUID":"o1","weight":1.0}
  ]
}"#;

/// Re-entrant creature the forward-only guard must reject.
const RECURRENT: &str = r#"{
  "semanticVersion":"4.0.0","forwardOnly":false,"input":1,"output":1,
  "neurons":[{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
  "synapses":[{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
}"#;

/// A one-record corpus mapping `input=1.0` onto `target=2.0`.
fn corpus() -> (TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let mut file = fs::File::create(data.join("0.bin")).unwrap();
    file.write_all(&1.0f32.to_le_bytes()).unwrap();
    file.write_all(&2.0f32.to_le_bytes()).unwrap();
    (dir, data)
}

/// Call the ABI exactly as a foreign caller does and hand back status + payload.
fn call_train(request: &str) -> (i32, String) {
    let bytes = request.as_bytes();
    let mut out = NeatBackpropBuffer {
        data: std::ptr::null_mut(),
        len: 0,
        capacity: 0,
    };
    // SAFETY: `bytes` outlives the call and `out` is a live, writable buffer
    // that owns nothing yet.
    let status = unsafe { neat_backprop_train(bytes.as_ptr(), bytes.len(), &raw mut out) };
    assert!(
        !out.data.is_null(),
        "every call must populate the out buffer"
    );
    // SAFETY: the library reported `out.len` readable bytes at `out.data`.
    let payload = unsafe { std::slice::from_raw_parts(out.data, out.len) }.to_vec();
    // SAFETY: `out` was produced by this library and has not been freed.
    unsafe { neat_backprop_buffer_free(&raw mut out) };
    assert!(out.data.is_null(), "free must reset the buffer to empty");
    assert_eq!(out.len, 0);
    assert_eq!(out.capacity, 0);
    (
        status,
        String::from_utf8(payload).expect("payload is UTF-8"),
    )
}

/// The base request every test starts from.
fn request(creature: &str, data: &Path, out_dir: &Path) -> Value {
    json!({
        "creatureJson": creature,
        "trainingData": data,
        "outputDir": out_dir,
        "epochs": 1,
        "maxRecords": 1,
        "seed": 7,
    })
}

#[test]
fn train_round_trips_a_tiny_creature_and_bin_dir() {
    let (dir, data) = corpus();
    let out_dir = dir.path().join("out");
    let (status, payload) = call_train(&request(CHAIN, &data, &out_dir).to_string());
    assert_eq!(status, NEAT_BACKPROP_OK, "train failed: {payload}");

    let response: TrainAbiResponse = serde_json::from_str(&payload).unwrap();
    assert_eq!(response.abi_version, NEAT_BACKPROP_ABI_VERSION);
    assert_eq!(response.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(response.epochs, 1);
    assert!(response.baseline_mse > 0.0);
    assert!(response.best_mse <= response.baseline_mse);
    assert!(response.baseline_score.is_none());
    assert!(response.best_score.is_none());
    assert!(response.best_trace_path.is_none());
    assert!(response.failed_trace_dir.is_none());

    // The returned creature JSON is the wire product: it must parse and be the
    // same bytes the run wrote to disk.
    let best: Value = serde_json::from_str(&response.best_creature_json).unwrap();
    assert_eq!(best["input"], json!(1));
    assert_eq!(best["output"], json!(1));
    assert_eq!(
        fs::read_to_string(&response.best_path).unwrap(),
        response.best_creature_json
    );
    assert_eq!(response.best_path, out_dir.join("best.json"));
    assert_eq!(response.journal_path, out_dir.join("journal.jsonl"));
    let journal = fs::read_to_string(&response.journal_path).unwrap();
    assert!(journal.contains("\"kind\":\"runHeader\""));
}

/// The ABI must reach the same learning as the CLI `train` path, not a
/// weaker in-process variant — an accepted epoch has to lower the MSE.
#[test]
fn train_accepts_an_improving_epoch() {
    let (dir, data) = corpus();
    let out_dir = dir.path().join("out");
    let mut req = request(CHAIN, &data, &out_dir);
    req["epochs"] = json!(4);
    req["learningRate"] = json!(0.5);
    let (status, payload) = call_train(&req.to_string());
    assert_eq!(status, NEAT_BACKPROP_OK, "train failed: {payload}");

    let response: TrainAbiResponse = serde_json::from_str(&payload).unwrap();
    assert!(
        response.accepted_epochs > 0,
        "a learnable corpus must accept at least one epoch"
    );
    assert!(
        response.best_mse < response.baseline_mse,
        "accepted epochs must lower MSE: {} -> {}",
        response.baseline_mse,
        response.best_mse
    );
}

/// `traceStore` must be reachable from the ABI, not only from the CLI (#78).
#[test]
fn train_reports_trace_artifacts() {
    let (dir, data) = corpus();
    let out_dir = dir.path().join("out");
    let store = dir.path().join("traces");
    let mut req = request(CHAIN, &data, &out_dir);
    req["traceStore"] = json!(store);
    req["learningRate"] = json!(0.5);
    let (status, payload) = call_train(&req.to_string());
    assert_eq!(status, NEAT_BACKPROP_OK, "train failed: {payload}");

    let response: TrainAbiResponse = serde_json::from_str(&payload).unwrap();
    let best_trace = response
        .best_trace_path
        .expect("an improving epoch writes a best trace");
    assert_eq!(best_trace, out_dir.join("best-trace.json"));
    let trace: Value = serde_json::from_str(&fs::read_to_string(&best_trace).unwrap()).unwrap();
    assert!(trace["neurons"].is_array(), "trace carries neurons");
    assert!(
        response.failed_trace_dir.is_none(),
        "an accepted-only run writes no failed candidates"
    );
}

/// A rejected epoch's candidate lands in the failed-trace store, and the ABI
/// has to hand that directory back rather than leave the caller to guess it.
#[test]
fn train_reports_the_failed_trace_store() {
    let (dir, data) = corpus();
    let out_dir = dir.path().join("out");
    let store = dir.path().join("traces");
    let mut req = request(CHAIN, &data, &out_dir);
    req["traceStore"] = json!(store);
    // Both genes propose as if the other holds still, so a rate of 1.0 takes
    // the whole step twice over and overshoots — a guaranteed rejection.
    req["learningRate"] = json!(1.0);
    req["stepScale"] = json!(1.0);
    req["maxBacktracks"] = json!(0);
    let (status, payload) = call_train(&req.to_string());
    assert_eq!(status, NEAT_BACKPROP_OK, "train failed: {payload}");

    let response: TrainAbiResponse = serde_json::from_str(&payload).unwrap();
    assert_eq!(response.accepted_epochs, 0);
    assert_eq!(response.failed_trace_dir, Some(store.join("failed")));
    assert!(
        store.join("failed").join("epoch-1.json").is_file(),
        "the rejected candidate must be captured"
    );
    assert!(response.best_trace_path.is_none());
}

/// Applying nothing must still be honest: `outputsOnly` + `hiddenOnly`
/// together select no genes, so no epoch can be accepted.
#[test]
fn apply_filters_are_forwarded() {
    let (dir, data) = corpus();
    let out_dir = dir.path().join("out");
    let mut req = request(CHAIN, &data, &out_dir);
    req["outputsOnly"] = json!(true);
    req["hiddenOnly"] = json!(true);
    req["learningRate"] = json!(0.5);
    let (status, payload) = call_train(&req.to_string());
    assert_eq!(status, NEAT_BACKPROP_OK, "train failed: {payload}");

    let response: TrainAbiResponse = serde_json::from_str(&payload).unwrap();
    assert_eq!(response.accepted_epochs, 0);
    assert!((response.best_mse - response.baseline_mse).abs() < 1e-12);
}

/// The whole-creature update budget is reachable over the ABI, not only from
/// the CLI (#109) — a tight budget must bind the in-process trainer too.
#[test]
fn the_trust_region_budget_is_forwarded() {
    let (dir, data) = corpus();
    let out_dir = dir.path().join("out");
    let mut req = request(CHAIN, &data, &out_dir);
    req["learningRate"] = json!(0.5);
    req["trustRegion"] = json!({ "l2": 1e-5 });
    let (status, payload) = call_train(&req.to_string());
    assert_eq!(status, NEAT_BACKPROP_OK, "train failed: {payload}");

    let journal = fs::read_to_string(out_dir.join("journal.jsonl")).unwrap();
    let header: Value = serde_json::from_str(journal.lines().next().unwrap()).unwrap();
    assert_eq!(header["trustRegion"]["l2"], json!(1e-5));
    let epoch: Value = serde_json::from_str(
        journal
            .lines()
            .find(|line| line.contains("\"kind\":\"epoch\""))
            .expect("epoch line"),
    )
    .unwrap();
    let realised = epoch["update"]["total"]["l2"].as_f64().expect("update L2");
    assert!(
        realised <= 1e-5 * 1.000_001,
        "budget not enforced: {realised}"
    );
    assert!(
        epoch["realisedStepScale"].as_f64().unwrap() < epoch["stepScale"].as_f64().unwrap(),
        "a bound update must journal a smaller realised step"
    );
}

/// An unusable budget must fail the call, not train with it ignored.
#[test]
fn an_unusable_trust_region_budget_fails_the_call() {
    let (dir, data) = corpus();
    let out_dir = dir.path().join("out");
    let mut req = request(CHAIN, &data, &out_dir);
    req["trustRegion"] = json!({ "l2": 0.0 });
    let (status, payload) = call_train(&req.to_string());
    assert_eq!(status, NEAT_BACKPROP_ERR_TRAIN_FAILED, "{payload}");
    assert!(payload.contains("trust-region"), "unexpected: {payload}");
}

#[test]
fn malformed_request_json_reports_an_invalid_argument() {
    let (status, payload) = call_train("{ not json");
    assert_eq!(status, NEAT_BACKPROP_ERR_INVALID_ARGUMENT);
    assert!(
        payload.starts_with("invalid train request JSON"),
        "unexpected message: {payload}"
    );
}

#[test]
fn a_recurrent_creature_fails_loudly() {
    let (dir, data) = corpus();
    let out_dir = dir.path().join("out");
    let (status, payload) = call_train(&request(RECURRENT, &data, &out_dir).to_string());
    assert_eq!(status, NEAT_BACKPROP_ERR_TRAIN_FAILED);
    assert!(
        payload.contains("forward-only"),
        "unexpected message: {payload}"
    );
    assert!(
        !out_dir.join("best.json").exists(),
        "a rejected run must not leave a best creature behind"
    );
}

#[test]
fn a_missing_training_directory_fails_loudly() {
    let (dir, _data) = corpus();
    let out_dir = dir.path().join("out");
    let missing = dir.path().join("no-such-corpus");
    let (status, payload) = call_train(&request(CHAIN, &missing, &out_dir).to_string());
    assert_eq!(status, NEAT_BACKPROP_ERR_TRAIN_FAILED);
    assert!(!payload.is_empty(), "a failure must carry a message");
}

#[test]
fn a_null_request_pointer_is_rejected_without_a_crash() {
    let mut out = NeatBackpropBuffer {
        data: std::ptr::null_mut(),
        len: 0,
        capacity: 0,
    };
    // SAFETY: a null request is the documented rejection path; `out` is live.
    let status = unsafe { neat_backprop_train(std::ptr::null(), 0, &raw mut out) };
    assert_eq!(status, NEAT_BACKPROP_ERR_INVALID_ARGUMENT);
    // SAFETY: the library reported `out.len` readable bytes at `out.data`.
    let message = unsafe { std::slice::from_raw_parts(out.data, out.len) }.to_vec();
    assert_eq!(
        String::from_utf8(message).unwrap(),
        "request pointer is null"
    );
    // SAFETY: `out` was produced by this library and has not been freed.
    unsafe { neat_backprop_buffer_free(&raw mut out) };
}

#[test]
fn a_null_out_pointer_is_rejected_without_a_crash() {
    let request = request(CHAIN, Path::new("data"), Path::new("out")).to_string();
    let bytes = request.as_bytes();
    // SAFETY: a null out buffer is the documented status-only rejection path.
    let status = unsafe { neat_backprop_train(bytes.as_ptr(), bytes.len(), std::ptr::null_mut()) };
    assert_eq!(status, NEAT_BACKPROP_ERR_INVALID_ARGUMENT);
}

#[test]
fn freeing_a_null_buffer_is_a_no_op() {
    // SAFETY: freeing null is documented as safe.
    unsafe { neat_backprop_buffer_free(std::ptr::null_mut()) };
}

#[test]
fn version_probes_match_the_crate() {
    assert_eq!(neat_backprop_abi_version(), NEAT_BACKPROP_ABI_VERSION);
    // SAFETY: the probe returns a static NUL-terminated string.
    let version = unsafe { CStr::from_ptr(neat_backprop_version()) };
    assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}
