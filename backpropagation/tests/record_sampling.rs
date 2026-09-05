//! Seeded record sampling for `--max-records` (issue #77).
//!
//! `--max-records N` used to mean "the first N records in directory scan
//! order", which over-fits the earliest files of a corpus. NEAT-AI's
//! TypeScript `selectFileSampleIndexes` instead draws the same *fraction*
//! from every file. These tests pin the ported contract end to end:
//!
//! * different `--seed` → a different record set;
//! * the same `--seed` → the identical record set and identical MSE;
//! * `--disable-random-samples` → a stable, non-shuffled per-file prefix;
//! * accumulate and the eval MSE inside one epoch see the identical records.

use neat_ai_backpropagation::sampling::{RecordSelection, plan_record_sample};
use neat_ai_backpropagation::{
    AcceptanceMode, ApplyOptions, BackpropConfig, TrainCreature, TrainRequest,
    accumulate_creature_learning_selected, compute_mse_selected, nearly_equal, run_train,
};
use neat_core::{TrainingDataConfig, compile_creature, parse_creature_json};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::Write;
use std::path::Path;
use tempfile::{TempDir, tempdir};

/// Single-input identity chain — the record's own value drives the error, so
/// which records were drawn is directly observable in the MSE.
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

/// Files in the fixture corpus.
const FILES: u64 = 4;
/// Records per fixture file.
const RECORDS_PER_FILE: u64 = 25;
/// Total records the fixture corpus holds.
const TOTAL_RECORDS: u64 = FILES * RECORDS_PER_FILE;

/// Four `.bin` files of 25 records each. The target grows with the global
/// record ordinal, so a sample skewed towards the early files scores a very
/// different MSE from one spread across the corpus.
fn corpus() -> TempDir {
    let dir = tempdir().unwrap();
    for file in 0..FILES {
        let mut f = std::fs::File::create(dir.path().join(format!("{file}.bin"))).unwrap();
        for i in 0..RECORDS_PER_FILE {
            let ordinal = file * RECORDS_PER_FILE + i;
            f.write_all(&1.0f32.to_le_bytes()).unwrap();
            f.write_all(&(ordinal as f32).to_le_bytes()).unwrap();
        }
    }
    dir
}

/// Write `creature.json` into `dir` and hand back its path.
fn creature_file(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("creature.json");
    std::fs::write(&path, CHAIN).unwrap();
    path
}

/// Baseline MSE of one `train` run over the fixture corpus.
fn train_baseline_mse(
    creature: &Path,
    data: &Path,
    out: &Path,
    seed: u64,
    disable_random_samples: bool,
) -> f64 {
    run_train(TrainRequest {
        creature: TrainCreature::Path(creature),
        training_data: data,
        config: &BackpropConfig::default(),
        epochs: 1,
        max_records: Some(20),
        seed,
        output_dir: out,
        scorer: None,
        apply: ApplyOptions::default(),
        acceptance: AcceptanceMode::Mse,
        accept_always: false,
        max_backtracks: 0,
        trace_store: None,
        disable_random_samples,
    })
    .unwrap()
    .baseline_mse
}

#[test]
fn different_seeds_select_different_records() {
    let dir = tempdir().unwrap();
    let data = corpus();
    let creature = creature_file(dir.path());

    let a = train_baseline_mse(&creature, data.path(), &dir.path().join("a"), 1, false);
    let b = train_baseline_mse(&creature, data.path(), &dir.path().join("b"), 99, false);

    assert!(
        !nearly_equal(a, b),
        "two seeds drew the same records — seed 1 MSE {a}, seed 99 MSE {b}"
    );
}

#[test]
fn the_same_seed_selects_the_same_records() {
    let dir = tempdir().unwrap();
    let data = corpus();
    let creature = creature_file(dir.path());

    let a = train_baseline_mse(&creature, data.path(), &dir.path().join("a"), 7, false);
    let b = train_baseline_mse(&creature, data.path(), &dir.path().join("b"), 7, false);

    assert!(
        nearly_equal(a, b),
        "seed 7 was not reproducible — {a} then {b}"
    );
}

#[test]
fn disabling_random_samples_is_a_stable_per_file_prefix() {
    let data = corpus();
    let cfg = TrainingDataConfig::new(1, 1);

    let first = plan_record_sample(data.path(), &cfg, 20, 1, true).unwrap();
    // A different seed must not move a non-shuffled selection.
    let second = plan_record_sample(data.path(), &cfg, 20, 4242, true).unwrap();

    for (a, b) in first.files().iter().zip(second.files()) {
        assert_eq!(a.path, b.path);
        assert_eq!(a.indexes, b.indexes, "seed moved a disabled-random sample");
        // ceil(25 * 20/100) = 5 records, taken as the file's leading prefix.
        assert_eq!(a.indexes, vec![0, 1, 2, 3, 4]);
    }
    assert_eq!(first.files().len(), FILES as usize);
    assert_eq!(first.total_records(), TOTAL_RECORDS);
}

#[test]
fn a_random_sample_spans_every_file_and_stays_ascending() {
    let data = corpus();
    let cfg = TrainingDataConfig::new(1, 1);
    let sample = plan_record_sample(data.path(), &cfg, 20, 11, false).unwrap();

    assert_eq!(
        sample.files().len(),
        FILES as usize,
        "a capped sample must still touch every file"
    );
    let mut shuffled_somewhere = false;
    for file in sample.files() {
        assert_eq!(file.indexes.len(), 5, "ceil(25 * 0.2) records per file");
        assert!(
            file.indexes.windows(2).all(|w| w[0] < w[1]),
            "indexes must be ascending for sequential reads: {:?}",
            file.indexes
        );
        assert!(file.indexes.iter().all(|i| *i < RECORDS_PER_FILE));
        if file.indexes != vec![0, 1, 2, 3, 4] {
            shuffled_somewhere = true;
        }
    }
    assert!(
        shuffled_somewhere,
        "the default sample must not be the deterministic prefix"
    );
    assert_eq!(sample.selected(), 20);
}

#[test]
fn a_cap_at_or_above_the_corpus_selects_everything() {
    let data = corpus();
    let cfg = TrainingDataConfig::new(1, 1);
    let sample = plan_record_sample(data.path(), &cfg, TOTAL_RECORDS + 5, 3, false).unwrap();
    assert_eq!(sample.selected(), TOTAL_RECORDS);
}

#[test]
fn accumulate_and_eval_mse_see_the_identical_sample() {
    let data = corpus();
    let creature = parse_creature_json(CHAIN).unwrap();
    let cfg = TrainingDataConfig::new(creature.input, creature.output);
    let sample = plan_record_sample(data.path(), &cfg, 20, 5, false).unwrap();
    let selection = RecordSelection::Sample(&sample);

    let mut eval_network = compile_creature(&creature).unwrap();
    let (eval_mse, eval_records) =
        compute_mse_selected(&creature, &mut eval_network, data.path(), selection).unwrap();

    let mut accumulate_network = compile_creature(&creature).unwrap();
    let mut rng = StdRng::seed_from_u64(5);
    let report = accumulate_creature_learning_selected(
        &creature,
        &mut accumulate_network,
        data.path(),
        &BackpropConfig::default(),
        selection,
        &mut rng,
    )
    .unwrap();

    assert_eq!(eval_records, 20);
    assert_eq!(
        eval_records, report.records,
        "the two surfaces scored different record counts"
    );
    assert!(
        nearly_equal(eval_mse, report.mse),
        "the two surfaces disagree on the sampled MSE — eval {eval_mse:.17e}, \
         accumulate {:.17e}",
        report.mse
    );
}

#[test]
fn an_empty_corpus_fails_loud() {
    let dir = tempdir().unwrap();
    let cfg = TrainingDataConfig::new(1, 1);
    let err = plan_record_sample(dir.path(), &cfg, 10, 1, false).unwrap_err();
    assert!(
        err.contains("no training records"),
        "expected a loud empty-corpus error, got: {err}"
    );
}
