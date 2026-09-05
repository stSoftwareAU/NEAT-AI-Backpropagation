//! Issue #92: `input < 1` / `output < 1` is never accepted — not on load, not
//! on write, not over the C ABI.
//!
//! The top-level `input` / `output` integers are the observation width and
//! cannot be re-derived (`neurons` lists only non-input neurons). A source
//! that arrives without them must fail before any epoch runs and leave no
//! `best.json`; a valid source must round-trip the width byte-identically
//! into `best.json`.

use neat_ai_backpropagation::backprop::{ApplyOptions, BackpropConfig};
use neat_ai_backpropagation::compare::run_compare;
use neat_ai_backpropagation::creature_io::{ObservationWidth, load_forward_only_creature};
use neat_ai_backpropagation::ffi::{NEAT_BACKPROP_ERR_TRAIN_FAILED, train_from_json};
use neat_ai_backpropagation::gradient_check::{GradientCheckRequest, run_gradient_check};
use neat_ai_backpropagation::sweep::{SweepRequest, run_sweep};
use neat_ai_backpropagation::train::{
    AcceptanceMode, TrainCreature, TrainRequest, TrainResult, run_train,
};
use neat_ai_backpropagation::trust_region::TrustRegion;
use neat_core::parse_creature_json;
use serde_json::{Value, json};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// Identity chain: `input-0 → h1 → o1`, one input, one output.
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

/// The same graph with the observation count zeroed — the case GRQ's wrapper
/// produced when it lost the width by scanning typed neurons.
const INPUT_ZERO: &str = r#"{
  "semanticVersion":"4.0.0","forwardOnly":true,"input":0,"output":1,
  "neurons":[
    {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
    {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
  ],
  "synapses":[
    {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
    {"fromUUID":"h1","toUUID":"o1","weight":1.0}
  ]
}"#;

/// `output: 0` while an output neuron is still present — the count, not the
/// neuron list, is authoritative.
const OUTPUT_ZERO: &str = r#"{
  "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":0,
  "neurons":[
    {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
    {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
  ],
  "synapses":[
    {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
    {"fromUUID":"h1","toUUID":"o1","weight":1.0}
  ]
}"#;

const INPUT_ERR: &str = "Must have at least one input neurons was: 0";
const OUTPUT_ERR: &str = "Must have at least one output neurons was: 0";

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

fn train(creature_path: &Path, data: &Path, out: &Path) -> Result<TrainResult, String> {
    run_train(TrainRequest {
        creature: TrainCreature::Path(creature_path),
        training_data: data,
        config: &BackpropConfig::default(),
        epochs: 2,
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
        trace_store: None,
    })
}

/// Nothing a `train` run writes may exist after a rejected load.
fn assert_no_run_artifacts(out: &Path) {
    for name in ["best.json", "journal.jsonl", "candidate.json"] {
        assert!(
            !out.join(name).exists(),
            "{name} must not be written for a rejected creature"
        );
    }
}

#[test]
fn train_rejects_input_zero_before_any_epoch() {
    let (dir, creature_path, data) = fixture(INPUT_ZERO);
    let out = dir.path().join("out");
    let err = train(&creature_path, &data, &out).expect_err("input 0 must be rejected");
    assert_eq!(err, INPUT_ERR);
    assert_no_run_artifacts(&out);
}

#[test]
fn train_rejects_output_zero_before_any_epoch() {
    let (dir, creature_path, data) = fixture(OUTPUT_ZERO);
    let out = dir.path().join("out");
    let err = train(&creature_path, &data, &out).expect_err("output 0 must be rejected");
    assert_eq!(err, OUTPUT_ERR);
    assert_no_run_artifacts(&out);
}

/// The width check runs before the forward-only check, so a widthless
/// recurrent creature reports the width — the more fundamental defect.
#[test]
fn width_is_checked_before_the_forward_only_rule() {
    let recurrent_widthless = INPUT_ZERO.replace("\"forwardOnly\":true", "\"forwardOnly\":false");
    let (_dir, creature_path, _data) = fixture(&recurrent_widthless);
    let err = load_forward_only_creature(&creature_path).unwrap_err();
    assert_eq!(err, INPUT_ERR);
}

#[test]
fn a_valid_source_round_trips_its_width_into_best_json() {
    let (dir, creature_path, data) = fixture(VALID);
    let out = dir.path().join("out");
    let result = train(&creature_path, &data, &out).unwrap();

    let source: Value = serde_json::from_str(VALID).unwrap();
    let best_text = fs::read_to_string(out.join("best.json")).unwrap();
    assert_eq!(
        best_text, result.best_json,
        "best.json is the returned text"
    );
    let best: Value = serde_json::from_str(&best_text).unwrap();
    assert_eq!(best["input"], source["input"]);
    assert_eq!(best["output"], source["output"]);
    // Byte-identical integers in the pretty-printed file, not just equal
    // after a lossy re-parse.
    assert!(best_text.contains("\"input\": 1"), "{best_text}");
    assert!(best_text.contains("\"output\": 1"), "{best_text}");
    assert_eq!(result.creature.input, 1);
    assert_eq!(result.creature.output, 1);
}

/// The CLI acceptance criterion: `train` on `{"input":0,...}` exits non-zero
/// with the width error and writes no `best.json`.
#[test]
fn cli_train_exits_non_zero_on_input_zero_and_writes_nothing() {
    let (dir, creature_path, data) = fixture(INPUT_ZERO);
    let out = dir.path().join("out");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_neat_ai_backpropagation"))
        .arg("train")
        .arg(&creature_path)
        .arg(&data)
        .arg("--epochs")
        .arg("1")
        .arg("--output-dir")
        .arg(&out)
        .output()
        .expect("run train");
    assert!(
        !output.status.success(),
        "train must exit non-zero on input 0"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(INPUT_ERR), "stderr: {stderr}");
    assert_no_run_artifacts(&out);
}

#[test]
fn cli_train_exits_non_zero_on_output_zero() {
    let (dir, creature_path, data) = fixture(OUTPUT_ZERO);
    let out = dir.path().join("out");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_neat_ai_backpropagation"))
        .arg("train")
        .arg(&creature_path)
        .arg(&data)
        .arg("--epochs")
        .arg("1")
        .arg("--output-dir")
        .arg(&out)
        .output()
        .expect("run train");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(OUTPUT_ERR), "stderr: {stderr}");
    assert_no_run_artifacts(&out);
}

/// The C ABI (issue #84) hands the creature over as JSON text; the same guard
/// must fire there and come back as a structured error, not a panic.
#[test]
fn ffi_train_rejects_a_widthless_creature() {
    let (dir, _creature_path, data) = fixture(INPUT_ZERO);
    let out = dir.path().join("out");
    let request = json!({
        "creatureJson": INPUT_ZERO,
        "trainingData": data,
        "outputDir": out,
        "epochs": 1,
        "maxRecords": 1,
        "seed": 7,
    });
    let err = train_from_json(&request.to_string()).expect_err("input 0 must be rejected");
    assert_eq!(err, INPUT_ERR);
    assert_no_run_artifacts(&out);
    // The raw ABI maps this to the train-failed status.
    let _ = NEAT_BACKPROP_ERR_TRAIN_FAILED;
}

#[test]
fn sibling_loaders_reject_a_widthless_creature() {
    let (dir, creature_path, data) = fixture(OUTPUT_ZERO);
    let cfg = BackpropConfig::default();

    let compare_out = dir.path().join("compare.json");
    let err = run_compare(&creature_path, &data, &cfg, Some(1), 1, &compare_out).unwrap_err();
    assert_eq!(err, OUTPUT_ERR);
    assert!(!compare_out.exists());

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
        facet_min_scored: 5,
        rank_limit: 5,
        output_dir: &dir.path().join("gc"),
    })
    .unwrap_err();
    assert_eq!(err, OUTPUT_ERR);

    let scales = [0.01];
    let sweep_out = dir.path().join("sweep");
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
        output_dir: &sweep_out,
    })
    .unwrap_err();
    assert_eq!(err, OUTPUT_ERR);
    assert!(!sweep_out.exists(), "sweep must not create its output dir");
}

/// The write guard: a creature whose width drifted from the source, or whose
/// width is < 1, is refused before any bytes are produced.
#[test]
fn write_guard_rejects_a_mismatched_or_widthless_creature() {
    let source = parse_creature_json(VALID).unwrap();
    let width = ObservationWidth::of(&source).unwrap();
    assert_eq!(
        width,
        ObservationWidth {
            input: 1,
            output: 1
        }
    );

    let mut drifted = source.clone();
    drifted.input = 2;
    let err = width.checked_json_pretty(&drifted).unwrap_err();
    assert!(err.contains("observation width changed"), "{err}");
    assert!(err.contains("written input=2 output=1"), "{err}");

    let mut zeroed = source.clone();
    zeroed.output = 0;
    let err = width.checked_json(&zeroed).unwrap_err();
    assert!(err.contains("observation width changed"), "{err}");

    // A zero source width is never a valid width to write against. neat-core
    // now enforces the same rule inside `parse_creature_json` (NEAT-AI-core
    // #550), so the widthless fixture no longer parses — assert that rejection
    // at the loader boundary, then exercise the local write guard against a
    // zero-width struct built by zeroing a parsed valid source.
    assert_eq!(
        parse_creature_json(INPUT_ZERO).unwrap_err().to_string(),
        INPUT_ERR,
        "the shared loader must refuse a widthless creature"
    );
    let mut zero_source = source.clone();
    zero_source.input = 0;
    assert_eq!(ObservationWidth::of(&zero_source).unwrap_err(), INPUT_ERR);
    let zero = ObservationWidth {
        input: 0,
        output: 1,
    };
    assert_eq!(zero.checked_json(&zero_source).unwrap_err(), INPUT_ERR);

    // The serialised-bytes check catches text that lost the width even when
    // the struct looked fine.
    let err = width
        .assert_written(r#"{"output":1,"neurons":[],"synapses":[]}"#)
        .unwrap_err();
    assert!(err.contains("`input` is missing"), "{err}");
    let err = width
        .assert_written(r#"{"input":1,"output":4,"neurons":[],"synapses":[]}"#)
        .unwrap_err();
    assert!(err.contains("serialised input=1 output=4"), "{err}");

    // And a faithful creature passes with the width intact in the text.
    let text = width.checked_json_pretty(&source).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["input"], 1);
    assert_eq!(value["output"], 1);
}
