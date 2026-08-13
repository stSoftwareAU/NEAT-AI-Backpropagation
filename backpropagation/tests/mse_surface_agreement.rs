//! Cross-surface MSE agreement (issue #34).
//!
//! This crate exposes two MSE surfaces that are meant to report the *same*
//! quantity — the per-record mean over outputs, averaged over records
//! (NEAT-AI `Costs.MSE`):
//!
//! * [`neat_ai_backpropagation::compute_mse`] — the eval path, which delegates
//!   to core's streaming/SIMD-batched helper;
//! * [`neat_ai_backpropagation::accumulate_creature_learning_report`] `.mse` —
//!   the fused accumulate pass behind the parity dumps, a strictly sequential
//!   scalar reduction over the same traced forward pass that feeds backprop.
//!
//! Nothing else ties them together, so they can drift silently. These tests are
//! that tie. Agreement is asserted with `nearly_equal` rather than bit
//! equality: the two summation orders differ by design (issue #30), so the
//! contract is "within tolerance", not "identical bits".

use neat_ai_backpropagation::{
    BackpropConfig, accumulate_creature_learning_report, compute_mse, nearly_equal,
};
use neat_core::{CreatureExport, compile_creature, parse_creature_json};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::Write;
use std::path::Path;
use tempfile::{TempDir, tempdir};

/// Plain feed-forward creature: two inputs, a `TANH` hidden layer, a
/// `LOGISTIC` output. All-standard squashes, so core's batch dispatch takes
/// the ordinary route.
const FEED_FORWARD: &str = r#"{
  "semanticVersion": "4.0.0",
  "forwardOnly": true,
  "input": 2,
  "output": 1,
  "neurons": [
    {"type":"hidden","uuid":"h1","bias":0.125,"squash":"TANH"},
    {"type":"hidden","uuid":"h2","bias":-0.25,"squash":"TANH"},
    {"type":"output","uuid":"o1","bias":0.5,"squash":"LOGISTIC"}
  ],
  "synapses": [
    {"fromUUID":"input-0","toUUID":"h1","weight":0.75},
    {"fromUUID":"input-1","toUUID":"h1","weight":-0.5},
    {"fromUUID":"input-0","toUUID":"h2","weight":0.25},
    {"fromUUID":"input-1","toUUID":"h2","weight":1.5},
    {"fromUUID":"h1","toUUID":"o1","weight":1.25},
    {"fromUUID":"h2","toUUID":"o1","weight":-0.75}
  ]
}"#;

/// Two hidden branches feeding a `MINIMUM` / `MAXIMUM` aggregate output.
/// `{SQUASH}` is substituted per fixture.
const AGGREGATE_OUTPUT: &str = r#"{
  "semanticVersion": "4.0.0",
  "forwardOnly": true,
  "input": 2,
  "output": 1,
  "neurons": [
    {"type":"hidden","uuid":"h1","bias":0.125,"squash":"IDENTITY"},
    {"type":"hidden","uuid":"h2","bias":-0.25,"squash":"IDENTITY"},
    {"type":"output","uuid":"o1","bias":0.5,"squash":"{SQUASH}"}
  ],
  "synapses": [
    {"fromUUID":"input-0","toUUID":"h1","weight":0.75},
    {"fromUUID":"input-1","toUUID":"h1","weight":-0.5},
    {"fromUUID":"input-0","toUUID":"h2","weight":0.25},
    {"fromUUID":"input-1","toUUID":"h2","weight":1.5},
    {"fromUUID":"h1","toUUID":"o1","weight":1.25},
    {"fromUUID":"h2","toUUID":"o1","weight":-0.75}
  ]
}"#;

/// Condition / positive / negative branches feeding an `IF` aggregate output.
const IF_OUTPUT: &str = r#"{
  "semanticVersion": "4.0.0",
  "forwardOnly": true,
  "input": 2,
  "output": 1,
  "neurons": [
    {"type":"hidden","uuid":"cond","bias":0.0,"squash":"IDENTITY"},
    {"type":"hidden","uuid":"pos","bias":0.125,"squash":"IDENTITY"},
    {"type":"hidden","uuid":"neg","bias":-0.125,"squash":"IDENTITY"},
    {"type":"output","uuid":"o1","bias":0.25,"squash":"IF"}
  ],
  "synapses": [
    {"fromUUID":"input-0","toUUID":"cond","weight":1.0},
    {"fromUUID":"input-1","toUUID":"cond","weight":-1.0},
    {"fromUUID":"input-0","toUUID":"pos","weight":2.0},
    {"fromUUID":"input-1","toUUID":"neg","weight":3.0},
    {"fromUUID":"cond","toUUID":"o1","weight":1.0,"type":"condition"},
    {"fromUUID":"pos","toUUID":"o1","weight":1.0,"type":"positive"},
    {"fromUUID":"neg","toUUID":"o1","weight":1.0,"type":"negative"}
  ]
}"#;

/// Records written to the fixture directory. Twelve is deliberately more than
/// eight so the corpus spans core's 8-way SIMD batch tier plus a scalar tail.
const RECORD_COUNT: u64 = 12;

/// Cap used for the `max_records` agreement check — both surfaces implement
/// the cap independently, and `5` lands mid-batch.
const CAPPED_RECORDS: u64 = 5;

/// Write `RECORD_COUNT` two-input / one-output records to a temp `.bin`
/// directory. Values are exact in `f32` and the condition input crosses zero,
/// so the `IF` fixture exercises both branches.
fn training_dir() -> TempDir {
    let dir = tempdir().unwrap();
    let mut f = std::fs::File::create(dir.path().join("0.bin")).unwrap();
    for i in 0..RECORD_COUNT {
        let in0 = 0.25f32 * i as f32 - 1.5;
        let in1 = 0.5f32 - 0.125f32 * i as f32;
        let target = 0.25f32 + 0.0625f32 * i as f32;
        for v in [in0, in1, target] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    dir
}

/// Both MSE surfaces for one creature over one corpus, under the same cap.
fn both_surfaces(
    creature: &CreatureExport,
    dir: &Path,
    max_records: Option<u64>,
) -> ((f64, u64), (f64, u64)) {
    let mut eval_network = compile_creature(creature).unwrap();
    let eval = compute_mse(creature, &mut eval_network, dir, max_records).unwrap();

    let mut accumulate_network = compile_creature(creature).unwrap();
    let config = BackpropConfig::default();
    let mut rng = StdRng::seed_from_u64(34);
    let report = accumulate_creature_learning_report(
        creature,
        &mut accumulate_network,
        dir,
        &config,
        max_records,
        &mut rng,
    )
    .unwrap();

    (eval, (report.mse, report.records))
}

/// Assert the two surfaces agree for `json`, both uncapped and under
/// `CAPPED_RECORDS`. `label` names the fixture in every failure message.
fn assert_surfaces_agree(label: &str, json: &str) {
    let dir = training_dir();
    let creature = parse_creature_json(json).unwrap();

    for cap in [None, Some(CAPPED_RECORDS)] {
        let ((eval_mse, eval_records), (accumulate_mse, accumulate_records)) =
            both_surfaces(&creature, dir.path(), cap);

        assert_eq!(
            eval_records, accumulate_records,
            "{label} (max_records {cap:?}): record counts disagree — \
             compute_mse scored {eval_records}, \
             accumulate_creature_learning_report scored {accumulate_records}"
        );
        assert_eq!(
            eval_records,
            cap.unwrap_or(RECORD_COUNT),
            "{label}: max_records {cap:?} was not honoured — scored {eval_records}"
        );
        assert!(
            nearly_equal(eval_mse, accumulate_mse),
            "{label} (max_records {cap:?}): MSE surfaces disagree — \
             compute_mse = {eval_mse:.17e}, \
             accumulate_creature_learning_report.mse = {accumulate_mse:.17e}, \
             difference = {:.17e}",
            (eval_mse - accumulate_mse).abs()
        );
    }
}

/// Guard the SIMD-tier coverage the fixtures depend on: shrinking the corpus
/// to eight records or fewer would silently stop exercising core's 8-way batch
/// tier, so fail loudly instead.
#[test]
fn fixture_corpus_exceeds_the_simd_batch_tier() {
    let dir = training_dir();
    let creature = parse_creature_json(FEED_FORWARD).unwrap();
    let mut network = compile_creature(&creature).unwrap();
    let (_, records) = compute_mse(&creature, &mut network, dir.path(), None).unwrap();
    assert!(
        records > 8,
        "fixture corpus must exceed core's 8-way SIMD batch tier, scored {records}"
    );
    const { assert!(CAPPED_RECORDS < RECORD_COUNT, "the cap must actually cap") };
}

#[test]
fn feed_forward_creature_agrees_across_both_mse_surfaces() {
    assert_surfaces_agree("feed-forward (TANH/LOGISTIC)", FEED_FORWARD);
}

#[test]
fn minimum_aggregate_creature_agrees_across_both_mse_surfaces() {
    assert_surfaces_agree(
        "aggregate MINIMUM",
        &AGGREGATE_OUTPUT.replace("{SQUASH}", "MINIMUM"),
    );
}

#[test]
fn maximum_aggregate_creature_agrees_across_both_mse_surfaces() {
    assert_surfaces_agree(
        "aggregate MAXIMUM",
        &AGGREGATE_OUTPUT.replace("{SQUASH}", "MAXIMUM"),
    );
}

#[test]
fn if_aggregate_creature_agrees_across_both_mse_surfaces() {
    assert_surfaces_agree("aggregate IF", IF_OUTPUT);
}
