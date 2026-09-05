//! Issue #78: `train --trace-store` writes NEAT-AI `CreatureTrace` artifacts.
//!
//! NEAT-AI's TypeScript trainer snapshots `creature.traceJSON()` into
//! `traceStore/failed/` whenever an iteration makes the network worse, and
//! keeps the best iteration's trace for `TrainingResult.trace`. The Rust
//! trainer already runs the traced forward pass — these tests pin that it now
//! serialises that state in the same UUID-keyed wire format:
//!
//! * a rejected epoch writes `<store>/failed/epoch-<N>.json`;
//! * an improving epoch writes `best-trace.json` beside `best.json`;
//! * without `--trace-store` nothing extra is written;
//! * the payload round-trips as NEAT-AI `CreatureTrace` — UUID endpoints on
//!   every gene and per-gene `trace` state.

use neat_ai_backpropagation::trace::{NeuronTraceState, SynapseTraceState};
use neat_ai_backpropagation::trust_region::TrustRegion;
use neat_ai_backpropagation::{
    AcceptanceMode, ApplyOptions, BackpropConfig, TrainCreature, TrainRequest, run_train,
};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// Identity chain: input 1 activates `h1` then `o1`, so the prediction is 1.
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

/// One-record corpus of `1 → target`, leaving a gap the epoch accumulates.
fn fixture(target: f32) -> (TempDir, PathBuf, PathBuf) {
    let dir = tempdir().unwrap();
    let creature = dir.path().join("creature.json");
    fs::write(&creature, CHAIN).unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let mut f = fs::File::create(data.join("0.bin")).unwrap();
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    f.write_all(&target.to_le_bytes()).unwrap();
    drop(f);
    (dir, creature, data)
}

/// `train` over the fixture with an optional trace store.
///
/// Both genes of the chain propose as if the other holds still, so a learning
/// rate of `1.0` takes the whole single-gene step twice over and lands well
/// past the target — a guaranteed rejection with real accumulated state. The
/// default `0.01` rate closes part of the gap and is accepted.
fn train(
    creature: &Path,
    data: &PathBuf,
    out: &PathBuf,
    trace_store: Option<&PathBuf>,
    learning_rate: f64,
) {
    let config = BackpropConfig {
        learning_rate,
        initial_learning_rate: learning_rate,
        ..BackpropConfig::default()
    };
    run_train(TrainRequest {
        creature: TrainCreature::Path(creature),
        training_data: data,
        config: &config,
        epochs: 1,
        max_records: Some(1),
        seed: 1,
        disable_random_samples: false,
        output_dir: out,
        scorer: None,
        apply: ApplyOptions::default(),
        acceptance: AcceptanceMode::Mse,
        accept_always: false,
        max_backtracks: 0,
        step_scale_ladder: &[],
        trust_region: TrustRegion::default(),
        trace_store: trace_store.map(PathBuf::as_path),
    })
    .unwrap();
}

/// Every neuron / synapse in the trace must carry its UUID endpoints, and the
/// genes the epoch actually accumulated must carry `trace` state.
fn assert_creature_trace(trace: &Value) {
    let neurons = trace["neurons"].as_array().expect("neurons array");
    assert_eq!(neurons.len(), 2, "trace keeps every export neuron");
    let uuids: Vec<&str> = neurons
        .iter()
        .map(|n| n["uuid"].as_str().expect("neuron uuid"))
        .collect();
    assert_eq!(uuids, vec!["h1", "o1"]);

    let traced: Vec<NeuronTraceState> = neurons
        .iter()
        .filter_map(|n| n.get("trace"))
        .map(|t| serde_json::from_value(t.clone()).expect("neuron trace state"))
        .collect();
    assert!(
        !traced.is_empty(),
        "an accumulated epoch must trace at least one neuron"
    );
    for state in &traced {
        assert!(state.count > 0.0, "traced neuron has a non-zero count");
        assert!(
            state.minimum_activation <= state.maximum_activation,
            "activation range is ordered: {state:?}"
        );
    }

    let synapses = trace["synapses"].as_array().expect("synapses array");
    assert_eq!(synapses.len(), 2, "trace keeps every export synapse");
    for s in synapses {
        assert!(s["fromUUID"].as_str().is_some(), "synapse fromUUID");
        assert!(s["toUUID"].as_str().is_some(), "synapse toUUID");
    }
    let traced: Vec<SynapseTraceState> = synapses
        .iter()
        .filter_map(|s| s.get("trace"))
        .map(|t| serde_json::from_value(t.clone()).expect("synapse trace state"))
        .collect();
    assert!(
        !traced.is_empty(),
        "an accumulated epoch must trace at least one synapse"
    );
    for state in &traced {
        assert!(state.count > 0.0, "traced synapse has a non-zero count");
    }

    // The export half of the payload is intact, so NEAT-AI can load the trace
    // as a creature as well as read its per-gene state.
    assert_eq!(trace["input"].as_u64(), Some(1));
    assert_eq!(trace["output"].as_u64(), Some(1));
    assert_eq!(trace["forwardOnly"].as_bool(), Some(true));
}

#[test]
fn rejected_epoch_writes_a_failed_trace_with_uuid_endpoints() {
    let (dir, creature, data) = fixture(1.5);
    let out = dir.path().join("out");
    let store = dir.path().join("trace-store");
    train(&creature, &data, &out, Some(&store), 1.0);

    let failed = store.join("failed").join("epoch-1.json");
    let text = fs::read_to_string(&failed)
        .unwrap_or_else(|e| panic!("rejected epoch writes {}: {e}", failed.display()));
    assert!(!text.trim().is_empty(), "trace artifact is not empty");
    assert_creature_trace(&serde_json::from_str::<Value>(&text).unwrap());

    // Nothing improved, so there is no best trace to keep.
    assert!(!out.join("best-trace.json").exists());
}

#[test]
fn improving_epoch_writes_the_best_trace_beside_best_json() {
    let (dir, creature, data) = fixture(2.0);
    let out = dir.path().join("out");
    let store = dir.path().join("trace-store");
    train(&creature, &data, &out, Some(&store), 0.01);

    let best_trace = out.join("best-trace.json");
    let text = fs::read_to_string(&best_trace)
        .unwrap_or_else(|e| panic!("improving epoch writes {}: {e}", best_trace.display()));
    assert_creature_trace(&serde_json::from_str::<Value>(&text).unwrap());
    assert!(
        out.join("best.json").is_file(),
        "the best trace sits beside best.json"
    );
    assert!(
        !store.join("failed").join("epoch-1.json").exists(),
        "an improving epoch is not a failed candidate"
    );
}

#[test]
fn without_a_trace_store_no_trace_artifact_is_written() {
    let (dir, creature, data) = fixture(1.5);
    let out = dir.path().join("out");
    train(&creature, &data, &out, None, 1.0);

    assert!(out.join("best.json").is_file(), "the run still completes");
    assert!(!out.join("best-trace.json").exists());
    assert!(!dir.path().join("trace-store").exists());
}

/// The store is the NEAT-AI `traceStore` directory — `train` creates it (and
/// the `failed/` subdirectory) rather than failing on a missing path.
#[test]
fn a_missing_trace_store_directory_is_created() {
    let (dir, creature, data) = fixture(1.5);
    let out = dir.path().join("out");
    let store = dir.path().join("nested").join("trace-store");
    assert!(!store.exists());
    train(&creature, &data, &out, Some(&store), 1.0);
    assert!(store.join("failed").is_dir());
}

/// Acceptance criterion: the flag is documented in `train --help`.
#[test]
fn train_help_documents_the_trace_store_flag() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_neat_ai_backpropagation"))
        .args(["train", "--help"])
        .output()
        .expect("run train --help");
    assert!(output.status.success(), "train --help exits cleanly");
    let help = String::from_utf8(output.stdout).expect("utf-8 help text");
    assert!(
        help.contains("--trace-store"),
        "train --help documents --trace-store:\n{help}"
    );
}
