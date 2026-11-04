//! Issue #214: an absolute deadline reaches the native trainer through both
//! entry points. The C ABI takes `deadlineEpochMs` and the CLI takes
//! `--deadline-epoch-ms`; each must stop the run cooperatively, keep the learning
//! it already accepted, and say so in the response or `result.json`.
//!
//! These tests call the exported `extern "C"` symbols and the built binary
//! directly. A missing library or binary fails the build or the test; nothing is
//! skipped.
//!
//! Why the productive fixture is sized as it is. The timings below are indicative
//! only, for a debug build: the baseline error scan takes about 0.2 s and one
//! accumulation pass about 24 times that (roughly 4 to 5 s). The assertions do
//! not depend on these figures beyond the 1 s deadline being shorter than one
//! accumulation pass. One epoch fits inside the 31 s budget (1 s deadline plus
//! 30 s grace), so the wall-clock bound alone cannot tell a
//! mid-epoch stop from a stop between epochs. `epochs: 100` makes the
//! uninterrupted request roughly eight minutes of work, far beyond the grace
//! window, so the bound does catch a run that ignores the deadline. The journal
//! evidence is what proves the cut happened inside epoch 1: exactly one epoch
//! line, marked partial, then a `timedOut` line, with `completedEpochs` at 0.

#![cfg(unix)]

use neat_ai_backpropagation::deadline::{TimeoutPhase, system_epoch_ms};
use neat_ai_backpropagation::ffi::{
    MeasureReport, NEAT_BACKPROP_ABI_VERSION, NEAT_BACKPROP_ERR_INVALID_ARGUMENT, NEAT_BACKPROP_OK,
    NeatBackpropBuffer, TRAIN_RESULT_FILE, TrainAbiResponse, neat_backprop_abi_version,
    neat_backprop_buffer_free, neat_backprop_capabilities, neat_backprop_train,
};
use neat_ai_backpropagation::nearly_equal;
use neat_core::{CreatureExport, parse_creature_json};
use serde_json::{Value, json};
use std::ffi::{CStr, OsStr};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::{TempDir, tempdir};

/// Wall-clock budget for a run, in milliseconds, measured from just before the call.
const DEADLINE_MS: u64 = 1000;

/// Time a run may take past its deadline before it counts as an overrun.
const GRACE_MS: u64 = 30_000;

/// Slack on top of deadline and grace before the test stops waiting for a result.
const OVERRUN_MARGIN_MS: u64 = 4_000;

/// Records in the productive corpus.
const RECORDS: u32 = 8_000;

/// Identity hidden neurons in the productive creature.
const HIDDEN: u32 = 500;

/// Records in the small contract-test corpus.
const SMALL_RECORDS: u32 = 100;

/// The CLI binary, built alongside this test. A missing binary fails the build.
const CLI: &str = env!("CARGO_BIN_EXE_neat_ai_backpropagation");

/// Identity chain `input-0 → h1 → h2 → o1`, the small fixture for the contract tests.
const CHAIN3: &str = r#"{
  "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
  "neurons":[
    {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
    {"type":"hidden","uuid":"h2","bias":0.0,"squash":"IDENTITY"},
    {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
  ],
  "synapses":[
    {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
    {"fromUUID":"h1","toUUID":"h2","weight":1.0},
    {"fromUUID":"h2","toUUID":"o1","weight":1.0}
  ]
}"#;

/// Writes `records` little-endian `(x, 2x)` f32 pairs into one `0.bin`, with `x`
/// cycling through `0.1 ..= 1.0`. Every input is positive, so one gradient
/// direction improves the whole corpus.
fn write_corpus(records: u32) -> TempDir {
    let dir = tempdir().unwrap();
    let mut file = fs::File::create(dir.path().join("0.bin")).unwrap();
    for i in 0..records {
        let x = (i % 10 + 1) as f32 * 0.1;
        file.write_all(&x.to_le_bytes()).unwrap();
        file.write_all(&(2.0 * x).to_le_bytes()).unwrap();
    }
    dir
}

/// Wide forward-only fan-in: `input-0 → h{i}` for every hidden neuron, then
/// `h{i} → o1` at weight `1/HIDDEN`, so the output is the input. Hidden neurons
/// come first, and synapses are sorted by target as neat-core requires.
fn productive_creature() -> String {
    let mut neurons: Vec<Value> = (0..HIDDEN)
        .map(|i| json!({"type": "hidden", "uuid": format!("h{i}"), "bias": 0.0, "squash": "IDENTITY"}))
        .collect();
    neurons.push(json!({"type": "output", "uuid": "o1", "bias": 0.0, "squash": "IDENTITY"}));
    let fan_in = (0..HIDDEN)
        .map(|i| json!({"fromUUID": "input-0", "toUUID": format!("h{i}"), "weight": 1.0}));
    let fan_out = (0..HIDDEN).map(
        |i| json!({"fromUUID": format!("h{i}"), "toUUID": "o1", "weight": 1.0 / f64::from(HIDDEN)}),
    );
    let synapses: Vec<Value> = fan_in.chain(fan_out).collect();
    json!({
        "semanticVersion": "4.0.0",
        "forwardOnly": true,
        "input": 1,
        "output": 1,
        "neurons": neurons,
        "synapses": synapses,
    })
    .to_string()
}

/// The productive fixture both entry points run: the wide creature over the full
/// corpus. The caller keeps the corpus alive for the run.
fn productive_fixture() -> (TempDir, String) {
    (write_corpus(RECORDS), productive_creature())
}

/// The productive creature as parsed input, for comparison with what comes back.
fn productive_input() -> CreatureExport {
    parse_creature_json(&productive_creature()).unwrap()
}

/// Every bias and weight, in a fixed order.
fn parameters(creature: &CreatureExport) -> Vec<f64> {
    creature
        .neurons
        .iter()
        .map(|n| n.bias)
        .chain(creature.synapses.iter().map(|s| s.weight))
        .collect()
}

/// Checks a returned creature keeps the input's neuron and synapse counts and
/// only finite parameters, and returns its parameters.
fn assert_returned_creature(text: &str, input: &CreatureExport) -> Vec<f64> {
    let returned = parse_creature_json(text).expect("the returned creature parses");
    assert_eq!(returned.neurons.len(), input.neurons.len(), "neuron count");
    assert_eq!(
        returned.synapses.len(),
        input.synapses.len(),
        "synapse count"
    );
    let values = parameters(&returned);
    assert!(
        values.iter().all(|v| v.is_finite()),
        "a returned bias or weight is not finite"
    );
    values
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

/// How long a run may take before the test calls it an overrun.
fn outer_bound() -> Duration {
    Duration::from_millis(DEADLINE_MS + GRACE_MS + OVERRUN_MARGIN_MS)
}

/// Runs `call_train` on its own thread and waits at most [`outer_bound`]. Returns
/// the status, the payload and the wall-clock time the call took.
fn call_train_bounded(request: String) -> (i32, String, Duration) {
    let (tx, rx) = mpsc::channel();
    let started = Instant::now();
    thread::spawn(move || {
        // A failed send means the test has already given up; nothing is left to report.
        let _ = tx.send(call_train(&request));
    });
    match rx.recv_timeout(outer_bound()) {
        Ok((status, payload)) => (status, payload, started.elapsed()),
        Err(RecvTimeoutError::Timeout) => panic!(
            "native call overran its deadline + grace: no return after {:?}",
            started.elapsed()
        ),
        Err(RecvTimeoutError::Disconnected) => {
            panic!("native call thread ended without a result (call_train panicked)")
        }
    }
}

/// Runs the CLI with stdin closed and both streams piped. The child is waited on
/// by a helper thread, so an overrun is killed rather than left running.
fn run_cli_bounded<S: AsRef<OsStr>>(args: &[S]) -> (Output, Duration) {
    let started = Instant::now();
    let child = Command::new(CLI)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the CLI binary");
    let pid = child.id();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });
    match rx.recv_timeout(outer_bound()) {
        Ok(output) => (output.expect("wait for the CLI"), started.elapsed()),
        Err(RecvTimeoutError::Timeout) => {
            // Best effort: the panic below reports the overrun either way.
            let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
            panic!(
                "native call overran its deadline + grace: CLI pid {pid} killed after {:?}",
                started.elapsed()
            );
        }
        Err(RecvTimeoutError::Disconnected) => {
            panic!("CLI wait thread ended without a result")
        }
    }
}

/// Runs the CLI to completion with stdin closed.
fn run_cli<S: AsRef<OsStr>>(args: &[S]) -> Output {
    Command::new(CLI)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("run the CLI binary")
}

/// `train` arguments for one run: the creature, the corpus, a fixed seed, the
/// output directory, then any extra flags.
fn train_args(creature: &Path, data: &Path, out: &Path, flags: &[&str]) -> Vec<String> {
    let mut args = vec![
        "train".to_string(),
        creature.display().to_string(),
        data.display().to_string(),
        "--seed".to_string(),
        "7".to_string(),
        "--output-dir".to_string(),
        out.display().to_string(),
    ];
    args.extend(flags.iter().map(ToString::to_string));
    args
}

/// Writes a creature JSON file into `dir` and returns its path.
fn write_creature(dir: &Path, creature: &str) -> PathBuf {
    let path = dir.join("creature.json");
    fs::write(&path, creature).unwrap();
    path
}

/// The journal's lines, parsed.
fn read_journal(out: &Path) -> Vec<Value> {
    fs::read_to_string(out.join("journal.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// The response `train` wrote to `result.json` in `out`.
fn read_result(out: &Path) -> TrainAbiResponse {
    let text = fs::read_to_string(out.join(TRAIN_RESULT_FILE)).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// A request that must succeed, parsed into its response.
fn train_ok(request: &Value) -> TrainAbiResponse {
    let (status, payload) = call_train(&request.to_string());
    assert_eq!(status, NEAT_BACKPROP_OK, "train failed: {payload}");
    serde_json::from_str(&payload).unwrap()
}

/// The small contract-test request: the chain over the small corpus, two epochs.
fn small_request(data: &Path, out: &Path) -> Value {
    json!({
        "creatureJson": CHAIN3,
        "trainingData": data,
        "outputDir": out,
        "epochs": 2,
        "seed": 7,
    })
}

/// The capability probe, parsed. The static C string is the document itself.
fn capabilities_probe() -> Value {
    let ptr = neat_backprop_capabilities();
    // SAFETY: the probe returns a static NUL-terminated string.
    let text = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
    serde_json::from_str(text).unwrap()
}

/// Asserts the run finished inside the deadline plus grace.
fn assert_within_deadline_and_grace(elapsed: Duration) {
    let limit = Duration::from_millis(DEADLINE_MS + GRACE_MS);
    assert!(
        elapsed < limit,
        "run took {elapsed:?}, over the {limit:?} deadline + grace"
    );
}

/// The contract every productive run meets: the deadline stopped it inside epoch
/// 1, the learning it kept is in the returned creature, and the measures say so.
/// Returns the records the partial verification covered.
fn assert_productive_response(
    response: &TrainAbiResponse,
    deadline_epoch_ms: u64,
    out: &Path,
) -> u64 {
    assert!(response.timed_out, "the deadline must cut the run short");
    let timeout = response
        .timeout
        .expect("a timed-out run carries its timeout record");
    assert_eq!(
        timeout.phase,
        TimeoutPhase::Accumulation,
        "expiry must land in accumulation"
    );
    assert_eq!(timeout.completed_epochs, 0, "no epoch completed");
    assert!(timeout.weights_changed, "the partial epoch was accepted");
    assert_eq!(
        timeout.deadline_epoch_ms, deadline_epoch_ms,
        "the deadline is echoed"
    );
    assert_eq!(response.completed_epochs, 0);
    assert_eq!(
        response.accepted_epochs, 1,
        "the partial epoch was accepted"
    );

    assert!(
        matches!(response.baseline_measure, MeasureReport::Full { .. }),
        "baseline was {:?}",
        response.baseline_measure
    );
    assert!(
        response.baseline_mse.is_some(),
        "a full baseline carries its mse"
    );
    let MeasureReport::Partial { records, .. } = response.best_measure else {
        panic!(
            "best must be a partial measure, was {:?}",
            response.best_measure
        );
    };
    assert_eq!(
        response.best_mse, None,
        "an incomplete figure is never a full-corpus score"
    );

    let input = productive_input();
    let returned = assert_returned_creature(&response.best_creature_json, &input);
    assert_ne!(
        returned,
        parameters(&input),
        "the kept learning must move the parameters"
    );

    let journal = read_journal(out);
    let epochs: Vec<&Value> = journal
        .iter()
        .filter(|line| line["kind"] == "epoch")
        .collect();
    assert_eq!(
        epochs.len(),
        1,
        "exactly one epoch line: the cut was inside epoch 1"
    );
    assert_eq!(epochs[0]["partial"], true, "the epoch line is partial");
    let journal_records = epochs[0]["records"]
        .as_u64()
        .expect("the epoch line carries records");
    assert!(
        journal_records >= 1 && journal_records < u64::from(RECORDS),
        "partial epoch covers {journal_records} records"
    );
    assert_eq!(
        journal.last().unwrap()["kind"],
        "timedOut",
        "the timedOut line closes the journal"
    );
    records
}

#[test]
fn ffi_deadline_stops_a_productive_first_epoch_mid_epoch() {
    let (data, creature) = productive_fixture();
    let out = tempdir().unwrap();

    let deadline = system_epoch_ms() + DEADLINE_MS;
    let request = json!({
        "creatureJson": creature,
        "trainingData": data.path(),
        "outputDir": out.path(),
        "epochs": 100,
        "learningRate": 0.01,
        "stepScale": 0.01,
        "maxBacktracks": 0,
        "seed": 7,
        "deadlineEpochMs": deadline,
    });
    let (status, payload, elapsed) = call_train_bounded(request.to_string());
    assert_eq!(status, NEAT_BACKPROP_OK, "train failed: {payload}");
    let response: TrainAbiResponse = serde_json::from_str(&payload).unwrap();

    let records = assert_productive_response(&response, deadline, out.path());
    assert_within_deadline_and_grace(elapsed);
    eprintln!(
        "ffi productive: elapsed {} ms, partial records {records}",
        elapsed.as_millis()
    );
}

#[test]
fn cli_deadline_stops_a_productive_first_epoch_mid_epoch() {
    let (data, creature) = productive_fixture();
    let work = tempdir().unwrap();
    let creature_path = write_creature(work.path(), &creature);
    let out = work.path().join("out");

    let deadline = system_epoch_ms() + DEADLINE_MS;
    let deadline_text = deadline.to_string();
    let args = train_args(
        &creature_path,
        data.path(),
        &out,
        &[
            "--epochs",
            "100",
            "--learning-rate",
            "0.01",
            "--step-scale",
            "0.01",
            "--max-backtracks",
            "0",
            "--deadline-epoch-ms",
            &deadline_text,
        ],
    );
    let (output, elapsed) = run_cli_bounded(&args);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exit {:?}: {stderr}",
        output.status
    );
    assert!(output.stdout.is_empty(), "train writes nothing to stdout");

    let response = read_result(&out);
    let records = assert_productive_response(&response, deadline, &out);
    assert_eq!(
        fs::read_to_string(out.join("best.json")).unwrap(),
        response.best_creature_json,
        "best.json is the returned creature"
    );
    assert_within_deadline_and_grace(elapsed);
    eprintln!(
        "cli productive: elapsed {} ms, partial records {records}",
        elapsed.as_millis()
    );
}

#[test]
fn ffi_omitted_deadline_runs_to_completion_with_full_measures() {
    let data = write_corpus(SMALL_RECORDS);
    let out = tempdir().unwrap();
    let response = train_ok(&small_request(data.path(), out.path()));

    assert!(!response.timed_out);
    assert!(response.timeout.is_none());
    assert_eq!(response.completed_epochs, 2);
    assert!(matches!(
        response.baseline_measure,
        MeasureReport::Full { .. }
    ));
    assert!(matches!(response.best_measure, MeasureReport::Full { .. }));
    assert!(response.baseline_mse.is_some());
    assert!(response.best_mse.is_some());
}

/// Two measures agree when they are the same state over the same records, with
/// MSEs equal to within float tolerance. A bounded run scores in 32-record batches
/// and an unbounded run in one pass, so the two sums can differ in the last bit.
fn assert_same_measure(actual: MeasureReport, expected: MeasureReport) {
    match (actual, expected) {
        (MeasureReport::Unmeasured, MeasureReport::Unmeasured) => {}
        (
            MeasureReport::Partial {
                mse: a,
                records: ra,
            },
            MeasureReport::Partial {
                mse: e,
                records: re,
            },
        )
        | (
            MeasureReport::Full {
                mse: a,
                records: ra,
            },
            MeasureReport::Full {
                mse: e,
                records: re,
            },
        ) => {
            assert_eq!(ra, re, "records scored");
            assert!(nearly_equal(a, e), "mse {a} vs {e}");
        }
        (actual, expected) => panic!("measure state {actual:?} vs {expected:?}"),
    }
}

/// Two optional MSEs agree when both are absent or both are equal to within
/// float tolerance.
fn assert_same_mse(actual: Option<f64>, expected: Option<f64>) {
    match (actual, expected) {
        (None, None) => {}
        (Some(a), Some(e)) => assert!(nearly_equal(a, e), "mse {a} vs {e}"),
        (actual, expected) => panic!("mse {actual:?} vs {expected:?}"),
    }
}

/// The outcome fields a deadline must not disturb when it never expires.
fn assert_same_outcome(actual: &TrainAbiResponse, expected: &TrainAbiResponse) {
    assert_eq!(actual.best_creature_json, expected.best_creature_json);
    assert_eq!(actual.timed_out, expected.timed_out);
    assert_eq!(actual.timeout, expected.timeout);
    assert_eq!(actual.completed_epochs, expected.completed_epochs);
    assert_eq!(actual.accepted_epochs, expected.accepted_epochs);
    assert_same_measure(actual.baseline_measure, expected.baseline_measure);
    assert_same_measure(actual.best_measure, expected.best_measure);
    assert_same_mse(actual.baseline_mse, expected.baseline_mse);
    assert_same_mse(actual.best_mse, expected.best_mse);
}

#[test]
fn ffi_null_deadline_matches_an_omitted_one() {
    let data = write_corpus(SMALL_RECORDS);
    let omitted_out = tempdir().unwrap();
    let omitted = train_ok(&small_request(data.path(), omitted_out.path()));

    let out = tempdir().unwrap();
    let mut request = small_request(data.path(), out.path());
    request["deadlineEpochMs"] = Value::Null;
    assert_same_outcome(&train_ok(&request), &omitted);
}

#[test]
fn ffi_far_future_deadline_matches_the_unbounded_run() {
    let data = write_corpus(SMALL_RECORDS);
    let unbounded_out = tempdir().unwrap();
    let unbounded = train_ok(&small_request(data.path(), unbounded_out.path()));

    let out = tempdir().unwrap();
    let mut request = small_request(data.path(), out.path());
    request["deadlineEpochMs"] = json!(system_epoch_ms() + 3_600_000);
    let bounded = train_ok(&request);

    assert!(!bounded.timed_out, "an hour away is not expired");
    assert_same_outcome(&bounded, &unbounded);
}

#[test]
fn ffi_expired_deadline_returns_the_input_unchanged_with_no_measurement() {
    let input = small_input();
    for deadline in [1_u64, 0] {
        let data = write_corpus(SMALL_RECORDS);
        let out = tempdir().unwrap();
        let mut request = small_request(data.path(), out.path());
        request["deadlineEpochMs"] = json!(deadline);
        let response = train_ok(&request);

        assert!(response.timed_out, "deadline {deadline} must expire");
        let timeout = response
            .timeout
            .expect("an expired run carries its timeout record");
        assert_eq!(timeout.phase, TimeoutPhase::Start, "deadline {deadline}");
        assert_eq!(timeout.completed_epochs, 0, "deadline {deadline}");
        assert!(!timeout.weights_changed, "deadline {deadline}");
        assert_eq!(timeout.deadline_epoch_ms, deadline, "echoed");
        assert_eq!(response.completed_epochs, 0);
        assert_eq!(response.accepted_epochs, 0);
        assert_eq!(response.baseline_measure, MeasureReport::Unmeasured);
        assert_eq!(response.best_measure, MeasureReport::Unmeasured);
        assert_eq!(response.baseline_mse, None);
        assert_eq!(response.best_mse, None);
        assert_eq!(
            assert_returned_creature(&response.best_creature_json, &input),
            parameters(&input),
            "deadline {deadline} must leave the creature unchanged"
        );
    }
}

/// The small contract fixture as parsed input, for the expired comparisons.
fn small_input() -> CreatureExport {
    parse_creature_json(CHAIN3).unwrap()
}

#[test]
fn ffi_malformed_deadline_is_an_invalid_argument() {
    let data = write_corpus(SMALL_RECORDS);
    let out = tempdir().unwrap();
    for value in [json!(-5), json!(1.5), json!("soon"), json!(true)] {
        let mut request = small_request(data.path(), out.path());
        request["deadlineEpochMs"] = value.clone();
        let (status, payload) = call_train(&request.to_string());

        assert_eq!(
            status, NEAT_BACKPROP_ERR_INVALID_ARGUMENT,
            "deadline {value}"
        );
        assert!(
            payload.starts_with("invalid train request JSON"),
            "unexpected message for {value}: {payload}"
        );
        assert!(
            payload.contains("deadlineEpochMs"),
            "message for {value} must name the field: {payload}"
        );
    }
}

#[test]
fn ffi_legacy_request_without_a_deadline_still_succeeds() {
    let data = write_corpus(SMALL_RECORDS);
    let out = tempdir().unwrap();
    // The request shape the ABI shipped with, before deadlines existed.
    let legacy = json!({
        "creatureJson": CHAIN3,
        "trainingData": data.path(),
        "outputDir": out.path(),
        "epochs": 1,
        "maxRecords": 1,
        "seed": 7,
    });
    let response = train_ok(&legacy);

    assert!(!response.timed_out);
    assert!(response.timeout.is_none());
    assert_eq!(response.completed_epochs, 1);
}

#[test]
fn ffi_capabilities_advertise_the_deadline() {
    let caps = capabilities_probe();
    assert_eq!(caps["abiVersion"], json!(neat_backprop_abi_version()));
    assert_eq!(caps["abiVersion"], json!(NEAT_BACKPROP_ABI_VERSION));
    assert_eq!(caps["capabilities"]["deadline"]["version"], json!(1));
    assert_eq!(
        caps["capabilities"]["deadline"]["requestField"],
        json!("deadlineEpochMs")
    );
}

#[test]
fn cli_omitted_deadline_runs_to_completion_with_full_measures() {
    let data = write_corpus(SMALL_RECORDS);
    let work = tempdir().unwrap();
    let creature = write_creature(work.path(), CHAIN3);
    let out = work.path().join("out");

    let output = run_cli(&train_args(
        &creature,
        data.path(),
        &out,
        &["--epochs", "2"],
    ));
    assert!(output.status.success(), "exit {:?}", output.status);
    assert!(output.stdout.is_empty(), "train writes nothing to stdout");

    let response = read_result(&out);
    assert!(!response.timed_out);
    assert!(response.timeout.is_none());
    assert_eq!(response.completed_epochs, 2);
    assert!(matches!(
        response.baseline_measure,
        MeasureReport::Full { .. }
    ));
    assert!(matches!(response.best_measure, MeasureReport::Full { .. }));
}

#[test]
fn cli_expired_deadline_leaves_the_creature_unchanged() {
    let data = write_corpus(SMALL_RECORDS);
    let work = tempdir().unwrap();
    let creature = write_creature(work.path(), CHAIN3);
    let out = work.path().join("out");

    let output = run_cli(&train_args(
        &creature,
        data.path(),
        &out,
        &["--deadline-epoch-ms", "1"],
    ));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exit {:?}: {stderr}",
        output.status
    );
    assert!(stderr.contains("timed out"), "stderr: {stderr}");

    let response = read_result(&out);
    assert!(response.timed_out);
    let timeout = response
        .timeout
        .expect("an expired run carries its timeout record");
    assert_eq!(timeout.phase, TimeoutPhase::Start);
    assert_eq!(response.baseline_measure, MeasureReport::Unmeasured);
    assert_eq!(response.best_measure, MeasureReport::Unmeasured);

    let best = fs::read_to_string(out.join("best.json")).unwrap();
    assert_eq!(
        assert_returned_creature(&best, &small_input()),
        parameters(&small_input()),
        "an expired CLI run must leave the creature unchanged"
    );
}

#[test]
fn cli_malformed_deadline_is_refused() {
    for value in ["--deadline-epoch-ms=-5", "--deadline-epoch-ms=abc"] {
        let output = run_cli(&["train", "creature.json", "data", value]);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success(),
            "{value} must be refused, exit {:?}",
            output.status
        );
        assert!(
            stderr.contains("--deadline-epoch-ms"),
            "stderr for {value} must name the flag: {stderr}"
        );
    }
}

#[test]
fn cli_capabilities_match_the_ffi_probe() {
    let output = run_cli(&["capabilities"]);
    assert!(output.status.success(), "exit {:?}", output.status);
    let stdout: Value =
        serde_json::from_slice(&output.stdout).expect("capabilities stdout is a JSON document");
    assert_eq!(stdout, capabilities_probe());
}
