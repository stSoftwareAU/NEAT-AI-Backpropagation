//! Issue #4073: a per-task wall-clock deadline must stop the `train` epoch
//! loop, keep everything already accepted, and journal the stop.
//!
//! The deadline is absolute (ms since the Unix epoch), so these tests pin it
//! with a deadline already in the past or far in the future rather than
//! sleeping against the wall clock. Each case is therefore deterministic.

use neat_ai_backpropagation::ffi::{TrainAbiRequest, TrainAbiResponse, train_from_json};
use neat_ai_backpropagation::train::TrainResult;
use neat_ai_backpropagation::trust_region::TrustRegion;
use neat_ai_backpropagation::{
    AcceptanceMode, ApplyOptions, BackpropConfig, TrainCreature, TrainRequest, run_train,
};
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
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

/// A deadline no test run reaches: 2100-01-01T00:00:00Z, in ms since the epoch.
const FAR_FUTURE_MS: u64 = 4_102_444_800_000;

/// A deadline that has already passed: 1 ms after the Unix epoch.
const PAST_MS: u64 = 1;

/// A two-record corpus in one `.bin` file, each record `(input, target)`.
fn corpus() -> (TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let mut file = std::fs::File::create(data.join("0.bin")).unwrap();
    for (input, target) in [(1.0f32, 2.0f32), (2.0, 4.0)] {
        file.write_all(&input.to_le_bytes()).unwrap();
        file.write_all(&target.to_le_bytes()).unwrap();
    }
    (dir, data)
}

/// Write the identity chain as `creature.json` and hand back its path.
fn creature_file(dir: &Path) -> PathBuf {
    let path = dir.join("creature.json");
    std::fs::write(&path, CHAIN).unwrap();
    path
}

/// Run the trainer over the corpus with the given epoch count and deadline.
fn train_with_deadline(
    creature: &Path,
    data: &Path,
    out: &Path,
    epochs: u64,
    deadline_epoch_ms: Option<u64>,
) -> TrainResult {
    run_train(TrainRequest {
        creature: TrainCreature::Path(creature),
        training_data: data,
        config: &BackpropConfig::default(),
        epochs,
        max_records: None,
        seed: 7,
        output_dir: out,
        scorer: None,
        apply: ApplyOptions::default(),
        acceptance: AcceptanceMode::Mse,
        accept_always: false,
        max_backtracks: 0,
        step_scale_ladder: &[],
        trust_region: TrustRegion::default(),
        trace_store: None,
        disable_random_samples: false,
        deadline_epoch_ms,
    })
    .unwrap()
}

/// Every line of `journal.jsonl` in `out`, in order.
fn read_journal(out: &Path) -> Vec<String> {
    std::fs::read_to_string(out.join("journal.jsonl"))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

/// How many journal lines carry the given `kind` discriminator.
fn kind_count(lines: &[String], kind: &str) -> usize {
    lines
        .iter()
        .filter(|line| serde_json::from_str::<Value>(line).unwrap()["kind"] == kind)
        .count()
}

#[test]
fn a_deadline_already_past_runs_no_epochs_and_keeps_best_json() {
    let (dir, data) = corpus();
    let creature = creature_file(dir.path());
    let out = dir.path().join("out");

    let result = train_with_deadline(&creature, &data, &out, 50, Some(PAST_MS));

    assert!(result.timed_out, "a past deadline must report timed_out");
    assert_eq!(result.accepted_epochs, 0);
    let journal = read_journal(&out);
    assert_eq!(
        kind_count(&journal, "epoch"),
        0,
        "no epoch may start once the deadline has passed: {journal:?}"
    );
    assert_eq!(
        journal.last().map(String::as_str),
        Some(r#"{"kind":"timedOut","completedEpochs":0,"deadlineEpochMs":1}"#),
        "the timedOut line must carry the exact contract shape"
    );
    assert!(
        out.join("best.json").is_file(),
        "the best-so-far creature must still be written"
    );
}

#[test]
fn a_deadline_far_in_the_future_runs_every_epoch() {
    let (dir, data) = corpus();
    let creature = creature_file(dir.path());
    let free_out = dir.path().join("free");
    let free = train_with_deadline(&creature, &data, &free_out, 3, None);
    let out = dir.path().join("out");

    let result = train_with_deadline(&creature, &data, &out, 3, Some(FAR_FUTURE_MS));

    assert!(
        !result.timed_out,
        "a far-future deadline must not stop the run"
    );
    assert_eq!(result.accepted_epochs, free.accepted_epochs);
    let journal = read_journal(&out);
    assert_eq!(
        kind_count(&journal, "timedOut"),
        0,
        "no timedOut line when the deadline was not hit: {journal:?}"
    );
    assert_eq!(
        kind_count(&journal, "epoch"),
        kind_count(&read_journal(&free_out), "epoch"),
        "a far-future deadline must run the same epochs as no deadline"
    );
}

#[test]
fn the_ffi_request_carries_the_deadline_and_rejects_unknown_fields() {
    let parsed: TrainAbiRequest = serde_json::from_value(json!({
        "creatureJson": CHAIN,
        "trainingData": "/unused",
        "outputDir": "/unused",
        "deadlineEpochMs": 1234,
    }))
    .unwrap();
    assert_eq!(parsed.deadline_epoch_ms, Some(1234));

    let null: TrainAbiRequest = serde_json::from_value(json!({
        "creatureJson": CHAIN,
        "trainingData": "/unused",
        "outputDir": "/unused",
        "deadlineEpochMs": null,
    }))
    .unwrap();
    assert_eq!(null.deadline_epoch_ms, None);

    let typo = train_from_json(
        &json!({
            "creatureJson": CHAIN,
            "trainingData": "/unused",
            "outputDir": "/unused",
            "deadlineEpochMsTypo": 1,
        })
        .to_string(),
    )
    .unwrap_err();
    assert!(typo.contains("unknown field"), "unexpected: {typo}");
}

#[test]
fn the_abi_reports_a_deadline_stop_as_timed_out() {
    let (dir, data) = corpus();
    let out = dir.path().join("out");
    let request = json!({
        "creatureJson": CHAIN,
        "trainingData": data,
        "outputDir": out,
        "epochs": 50,
        "seed": 7,
        "deadlineEpochMs": PAST_MS,
    });

    let payload = train_from_json(&request.to_string()).unwrap();
    let response: TrainAbiResponse = serde_json::from_str(&payload).unwrap();

    assert_eq!(response.abi_version, 2);
    assert!(response.timed_out, "unexpected response: {payload}");
    assert!(
        payload.contains(r#""timedOut":true"#),
        "unexpected: {payload}"
    );
}

#[test]
fn the_cli_accepts_a_deadline_flag_and_reports_the_stop() {
    let (dir, data) = corpus();
    let creature = creature_file(dir.path());
    let out = dir.path().join("out");

    let output = Command::new(env!("CARGO_BIN_EXE_neat_ai_backpropagation"))
        .arg("train")
        .arg(&creature)
        .arg(&data)
        .args(["--epochs", "5", "--deadline-epoch-ms", "1", "--output-dir"])
        .arg(&out)
        .stdin(Stdio::null())
        .output()
        .expect("run train");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "train failed: {stderr}");
    assert!(stderr.contains("timed_out=true"), "unexpected: {stderr}");
    assert_eq!(
        read_journal(&out).last().map(String::as_str),
        Some(r#"{"kind":"timedOut","completedEpochs":0,"deadlineEpochMs":1}"#),
    );
}

#[test]
fn train_help_lists_the_deadline_flag() {
    let output = Command::new(env!("CARGO_BIN_EXE_neat_ai_backpropagation"))
        .args(["train", "--help"])
        .stdin(Stdio::null())
        .output()
        .expect("run train --help");

    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("--deadline-epoch-ms"), "unexpected: {help}");
}
