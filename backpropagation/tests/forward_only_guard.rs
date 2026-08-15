//! Issue #54: every subcommand that feeds the accumulate engine must reject a
//! recurrent creature with the same error.
//!
//! `compare` had drifted — it parsed the creature and ran the unsupported
//! accumulate path silently, producing a parity dump instead of failing loudly
//! while `gradient-check`, `sweep`, and `train` all rejected the same file.

use neat_ai_backpropagation::backprop::{ApplyOptions, BackpropConfig};
use neat_ai_backpropagation::compare::run_compare;
use neat_ai_backpropagation::creature_io::load_forward_only_creature;
use neat_ai_backpropagation::gradient_check::{GradientCheckRequest, run_gradient_check};
use neat_ai_backpropagation::sweep::{SweepRequest, run_sweep};
use neat_ai_backpropagation::train::{TrainRequest, run_train};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tempfile::{TempDir, tempdir};

/// Identity chain plus a `o1 → h1` back edge, declared re-entrant.
const RECURRENT: &str = r#"{
  "semanticVersion":"4.0.0","forwardOnly":false,"input":1,"output":1,
  "neurons":[
    {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
    {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
  ],
  "synapses":[
    {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
    {"fromUUID":"h1","toUUID":"o1","weight":1.0},
    {"fromUUID":"o1","toUUID":"h1","weight":0.5}
  ]
}"#;

/// The same graph without the back edge, declared forward-only.
const FORWARD_ONLY: &str = r#"{
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

const EXPECTED: &str =
    "this trainer supports forward-only creatures only (no re-entrant / recurrent graphs)";

/// Write `creature` plus a one-record `.bin` corpus. Returns
/// `(dir, creature_path, data_dir)`.
fn fixture(creature: &str) -> (TempDir, PathBuf, PathBuf) {
    let dir = tempdir().unwrap();
    let creature_path = dir.path().join("creature.json");
    fs::write(&creature_path, creature).unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let mut f = fs::File::create(data.join("0.bin")).unwrap();
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    f.write_all(&2.0f32.to_le_bytes()).unwrap();
    drop(f);
    (dir, creature_path, data)
}

#[test]
fn compare_rejects_a_recurrent_creature() {
    let (dir, creature_path, data) = fixture(RECURRENT);
    let out = dir.path().join("compare.json");

    let err = run_compare(
        &creature_path,
        &data,
        &BackpropConfig::default(),
        Some(1),
        1,
        &out,
    )
    .expect_err("compare must reject a recurrent creature");

    assert_eq!(err, EXPECTED);
    assert!(!out.exists(), "no parity dump may be written on rejection");
}

#[test]
fn gradient_check_rejects_a_recurrent_creature() {
    let (dir, creature_path, data) = fixture(RECURRENT);
    let cfg = BackpropConfig::default();
    let err = run_gradient_check(GradientCheckRequest {
        creature: &creature_path,
        training_data: &data,
        config: &cfg,
        max_records: Some(1),
        seed: 1,
        sample_biases: 8,
        sample_weights: 8,
        fd_eps: 1e-4,
        step_scale: 1.0,
        outputs_only: false,
        hidden_only: false,
        output_dir: &dir.path().join("out"),
    })
    .expect_err("gradient-check must reject a recurrent creature");

    assert_eq!(err, EXPECTED);
}

#[test]
fn sweep_rejects_a_recurrent_creature() {
    let (dir, creature_path, data) = fixture(RECURRENT);
    let cfg = BackpropConfig::default();
    let scales = [0.01];
    let err = run_sweep(SweepRequest {
        creature: &creature_path,
        training_data: &data,
        eval_data: None,
        config: &cfg,
        max_records: Some(1),
        seed: 1,
        step_scales: &scales,
        outputs_only: false,
        hidden_only: false,
        skip_mse: false,
        output_dir: &dir.path().join("out"),
    })
    .expect_err("sweep must reject a recurrent creature");

    assert_eq!(err, EXPECTED);
}

#[test]
fn train_rejects_a_recurrent_creature() {
    let (dir, creature_path, data) = fixture(RECURRENT);
    let cfg = BackpropConfig::default();
    let err = run_train(TrainRequest {
        creature: &creature_path,
        training_data: &data,
        config: &cfg,
        epochs: 1,
        max_records: Some(1),
        seed: 1,
        output_dir: &dir.path().join("out"),
        scorer: None,
        apply: ApplyOptions::default(),
        accept_always: false,
        max_backtracks: 0,
    })
    .expect_err("train must reject a recurrent creature");

    assert_eq!(err, EXPECTED);
}

#[test]
fn loader_accepts_a_forward_only_creature() {
    let (_dir, creature_path, _data) = fixture(FORWARD_ONLY);
    let creature = load_forward_only_creature(&creature_path).unwrap();
    assert!(creature.forward_only);
    assert_eq!(creature.neurons.len(), 2);
}

#[test]
fn loader_reports_a_missing_file() {
    let dir = tempdir().unwrap();
    let err = load_forward_only_creature(&dir.path().join("absent.json"))
        .expect_err("a missing creature file must fail loudly");
    assert!(!err.is_empty());
    assert_ne!(err, EXPECTED);
}

#[test]
fn loader_reports_malformed_json() {
    let (_dir, creature_path, _data) = fixture("{ not json");
    let err = load_forward_only_creature(&creature_path)
        .expect_err("malformed creature JSON must fail loudly");
    assert!(!err.is_empty());
    assert_ne!(err, EXPECTED);
}
