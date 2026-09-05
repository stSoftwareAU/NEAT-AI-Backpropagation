//! Issue #94: every trained creature is validated by
//! `neat_core::creature_validate` before it is scored, written or returned.
//!
//! The failure this gate exists to stop is numeric: a gradient step that
//! produces a non-finite bias currently escapes as a "trained" creature. The
//! gate is at output, runs once per completed run, and refuses to write
//! anything when it fires.

use neat_ai_backpropagation::backprop::{ApplyOptions, BackpropConfig};
use neat_ai_backpropagation::sweep::{SweepRequest, run_sweep};
use neat_ai_backpropagation::train::{
    AcceptanceMode, TrainCreature, TrainRequest, TrainResult, run_train,
};
use neat_core::parse_creature_json;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// Identity chain `input-0 → h1 → o1`, one input, one output.
const VALID: &str = r#"{
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

/// Write the creature plus a one-record `.bin` corpus of `(observation,
/// target)` pairs. Returns `(dir, creature_path, data_dir)`.
fn fixture(records: &[(f32, f32)]) -> (TempDir, PathBuf, PathBuf) {
    let dir = tempdir().unwrap();
    let creature_path = dir.path().join("creature.json");
    fs::write(&creature_path, VALID).unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let mut f = fs::File::create(data.join("0.bin")).unwrap();
    for (observation, target) in records {
        f.write_all(&observation.to_le_bytes()).unwrap();
        f.write_all(&target.to_le_bytes()).unwrap();
    }
    drop(f);
    (dir, creature_path, data)
}

/// A configuration that diverges: an infinite L2 bias decay drives every
/// proposed bias to `-inf` on the first apply.
///
/// `neat-core` skips non-finite *gradients*, so a poisoned record cannot reach
/// the bias; a poisoned regularisation term can, and the config arrives from
/// the caller (CLI flags, or the FFI request JSON). This is the diverged run
/// the gate exists to catch.
fn diverging_config() -> BackpropConfig {
    BackpropConfig {
        l2_bias_decay: f64::INFINITY,
        ..BackpropConfig::default()
    }
}

fn train(
    creature_path: &Path,
    data: &Path,
    out: &Path,
    config: &BackpropConfig,
    accept_always: bool,
) -> Result<TrainResult, String> {
    run_train(TrainRequest {
        creature: TrainCreature::Path(creature_path),
        training_data: data,
        config,
        epochs: 2,
        max_records: None,
        seed: 1,
        disable_random_samples: true,
        output_dir: out,
        scorer: None,
        apply: ApplyOptions::default(),
        acceptance: AcceptanceMode::Mse,
        accept_always,
        max_backtracks: 0,
        step_scale_ladder: &[],
        trace_store: None,
    })
}

/// A run that diverged into a non-finite bias must fail loudly and leave no
/// `best.json` behind — the creature is neither returned nor written.
#[test]
fn train_refuses_to_return_a_creature_with_a_non_finite_bias() {
    // `--accept-always` is the production path that keeps a diverged
    // candidate: the MSE comparison that would otherwise roll it back is
    // `NaN < best`, which is false.
    let (dir, creature_path, data) = fixture(&[(1.0, 2.0), (2.0, 4.0)]);
    let out = dir.path().join("out");

    let err = train(&creature_path, &data, &out, &diverging_config(), true)
        .expect_err("a non-finite bias must not escape");

    assert!(
        err.contains("refusing to return a trained creature"),
        "{err}"
    );
    // The training run that produced it is named, along with neat-core's own
    // reason, message and offending neuron index.
    assert!(err.contains("train run in"), "{err}");
    assert!(err.contains("after 2 epochs"), "{err}");
    assert!(err.contains("bias"), "{err}");
    assert!(err.contains("neuron index"), "{err}");

    assert!(
        !out.join("best.json").exists(),
        "an invalid creature must not reach best.json"
    );
}

/// The gate is a check, not a transform: a healthy run still succeeds and
/// hands back exactly the creature and bytes it did before.
#[test]
fn a_healthy_train_run_passes_the_gate_unchanged() {
    let (dir, creature_path, data) = fixture(&[(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]);
    let out = dir.path().join("out");

    let result = train(
        &creature_path,
        &data,
        &out,
        &BackpropConfig::default(),
        false,
    )
    .expect("a healthy run must pass");

    // Same bytes on disk as in the result.
    let best_text = fs::read_to_string(out.join("best.json")).unwrap();
    assert_eq!(best_text, result.best_json);
    let written = parse_creature_json(&best_text).unwrap();

    // Topology preserved, every value finite, and the run actually improved.
    let source = parse_creature_json(VALID).unwrap();
    assert_eq!(written.input, source.input);
    assert_eq!(written.output, source.output);
    assert_eq!(written.neurons.len(), source.neurons.len());
    assert_eq!(written.synapses.len(), source.synapses.len());
    assert!(written.neurons.iter().all(|n| n.bias.is_finite()));
    assert!(written.synapses.iter().all(|s| s.weight.is_finite()));
    assert!(result.best_mse <= result.baseline_mse);

    // And the run is still deterministic under the gate: the same request
    // produces byte-identical output.
    let again = dir.path().join("again");
    let repeat = train(
        &creature_path,
        &data,
        &again,
        &BackpropConfig::default(),
        false,
    )
    .expect("a healthy run must pass");
    assert_eq!(repeat.best_json, result.best_json);
}

/// `sweep` writes a creature per step scale, so each one is gated as it is
/// produced — a diverged candidate never reaches `candidates/`.
#[test]
fn sweep_refuses_to_write_a_candidate_with_a_non_finite_bias() {
    let (dir, creature_path, data) = fixture(&[(1.0, 2.0), (2.0, 4.0)]);
    let out = dir.path().join("sweep");
    let scales = [0.01, 0.1];

    let err = run_sweep(SweepRequest {
        creature: &creature_path,
        training_data: &data,
        eval_data: None,
        config: &diverging_config(),
        max_records: None,
        seed: 1,
        step_scales: &scales,
        outputs_only: false,
        hidden_only: false,
        skip_mse: true,
        output_dir: &out,
    })
    .expect_err("a NaN candidate must not be written");

    assert!(
        err.contains("refusing to return a trained creature"),
        "{err}"
    );
    assert!(err.contains("sweep candidate st="), "{err}");
    assert!(
        !out.join("candidates").join("st0.01000000.json").exists(),
        "the invalid candidate must not reach disk"
    );
}
