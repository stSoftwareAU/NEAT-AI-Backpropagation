//! Issue #213: `run_train_until` stops cooperatively at an absolute deadline
//! inside an epoch, keeps the learning it already accepted, and returns useful
//! partial learning.
//!
//! Every phase is reached deterministically. A counting clock expires on its
//! `T`-th read, and [`first_tick`] scans `T` upward until a run stops in the
//! phase under test. Because the scan only lands on a tick where that phase's
//! own check exists, each test fails when only that check is removed.

use neat_ai_backpropagation::backprop::{ApplyOptions, BackpropConfig};
use neat_ai_backpropagation::deadline::{Deadline, TimeoutPhase};
use neat_ai_backpropagation::mse::compute_mse;
use neat_ai_backpropagation::train::{
    AcceptanceMode, TrainCreature, TrainEpochRecord, TrainRequest, TrainResult, TrainStatus,
    run_train, run_train_until,
};
use neat_ai_backpropagation::trust_region::TrustRegion;
use neat_ai_backpropagation::validate::TrainedTopology;
use neat_ai_backpropagation::{ErrorMeasure, nearly_equal};
use neat_core::{CreatureExport, compile_creature, parse_creature_json};
use serde_json::Value;
use std::cell::Cell;
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
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

/// Records in the corpus: 100, so every phase has several checks and the
/// baseline and candidate scans span more than one 32-record batch.
const RECORDS: u32 = 100;

/// Largest expiry tick the phase scans try before giving up.
const MAX_TICK: u64 = 5000;

/// Corpus of `(x, 2x)` pairs with `x` cycling through `0.1 ..= 1.0`. Every
/// input is positive, so one gradient direction improves the whole corpus.
fn corpus() -> TempDir {
    let dir = tempdir().unwrap();
    let mut f = fs::File::create(dir.path().join("0.bin")).unwrap();
    for i in 0..RECORDS {
        let x = (i % 10 + 1) as f32 * 0.1;
        f.write_all(&x.to_le_bytes()).unwrap();
        f.write_all(&(2.0 * x).to_le_bytes()).unwrap();
    }
    dir
}

/// The run settings every test varies from.
#[derive(Clone)]
struct Setup {
    epochs: u64,
    learning_rate: f64,
    apply: ApplyOptions,
    max_backtracks: u32,
    scorer: Option<PathBuf>,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            epochs: 1,
            learning_rate: 0.01,
            apply: ApplyOptions {
                step_scale: 0.01,
                ..ApplyOptions::default()
            },
            max_backtracks: 0,
            scorer: None,
        }
    }
}

/// One run's artefacts: the result, its journal lines, and the output
/// directory, kept alive so the files stay readable.
struct Run {
    result: TrainResult,
    journal: Vec<Value>,
    out: TempDir,
}

/// The backprop config a setup implies.
fn config_for(setup: &Setup) -> BackpropConfig {
    BackpropConfig {
        learning_rate: setup.learning_rate,
        initial_learning_rate: setup.learning_rate,
        ..BackpropConfig::default()
    }
}

/// The request both the bounded and the unbounded runs share.
fn request<'a>(
    data: &'a Path,
    out: &'a Path,
    config: &'a BackpropConfig,
    setup: &'a Setup,
) -> TrainRequest<'a> {
    TrainRequest {
        creature: TrainCreature::Json(VALID),
        training_data: data,
        config,
        epochs: setup.epochs,
        max_records: None,
        seed: 7,
        disable_random_samples: false,
        output_dir: out,
        scorer: setup.scorer.as_deref(),
        apply: setup.apply,
        trust_region: TrustRegion::default(),
        acceptance: AcceptanceMode::Mse,
        accept_always: false,
        max_backtracks: setup.max_backtracks,
        step_scale_ladder: &[],
        trace_store: None,
    }
}

fn read_journal(out: &Path) -> Vec<Value> {
    fs::read_to_string(out.join("journal.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// Run through [`run_train_until`] with the given deadline.
fn run_until(data: &Path, setup: &Setup, deadline: Option<&Deadline<'_>>) -> Run {
    let out = tempdir().unwrap();
    let config = config_for(setup);
    let result = run_train_until(request(data, out.path(), &config, setup), deadline).unwrap();
    let journal = read_journal(out.path());
    Run {
        result,
        journal,
        out,
    }
}

/// Run with a counting clock that expires on its `tick`-th read.
fn run_at_tick(data: &Path, setup: &Setup, tick: u64) -> Run {
    let reads = Cell::new(0u64);
    let deadline = Deadline::with_clock(tick, || {
        reads.set(reads.get() + 1);
        reads.get()
    });
    run_until(data, setup, Some(&deadline))
}

/// The first tick whose run satisfies `wanted`. Panics with a clear message
/// when no tick up to [`MAX_TICK`] does, which is what a removed check causes.
fn first_tick(
    phase: &str,
    data: &Path,
    setup: &Setup,
    wanted: impl Fn(&Run) -> bool,
) -> (u64, Run) {
    for tick in 1..=MAX_TICK {
        let run = run_at_tick(data, setup, tick);
        if wanted(&run) {
            return (tick, run);
        }
    }
    panic!("no expiry tick reaches phase {phase}");
}

/// Whether the run stopped in `phase`.
fn timed_out_in(phase: TimeoutPhase) -> impl Fn(&Run) -> bool {
    move |run| {
        matches!(
            run.result.status,
            TrainStatus::TimedOut { phase: p, .. } if p == phase
        )
    }
}

fn status_of(run: &Run) -> (TimeoutPhase, u64, bool) {
    match run.result.status {
        TrainStatus::TimedOut {
            phase,
            completed_epochs,
            weights_changed,
        } => (phase, completed_epochs, weights_changed),
        TrainStatus::Completed => panic!("expected a timed-out run"),
    }
}

fn epoch_lines(run: &Run) -> Vec<&Value> {
    run.journal
        .iter()
        .filter(|l| l["kind"] == "epoch")
        .collect()
}

/// Every weight and bias, in a fixed order.
fn parameters(creature: &CreatureExport) -> Vec<f64> {
    creature
        .neurons
        .iter()
        .map(|n| n.bias)
        .chain(creature.synapses.iter().map(|s| s.weight))
        .collect()
}

fn input_creature() -> CreatureExport {
    parse_creature_json(VALID).unwrap()
}

/// Corpus-wide mean squared error of `creature`.
fn corpus_mse(creature: &CreatureExport, data: &Path) -> f64 {
    let mut network = compile_creature(creature).unwrap();
    compute_mse(creature, &mut network, data, None).unwrap().0
}

/// Structural and numeric validity: the trainer never changes the genes, and
/// every parameter is finite.
fn assert_valid_topology(creature: &CreatureExport) {
    let input = input_creature();
    assert_eq!(creature.neurons.len(), input.neurons.len());
    assert_eq!(creature.synapses.len(), input.synapses.len());
    assert!(
        parameters(creature).iter().all(|v| v.is_finite()),
        "non-finite parameter in {creature:?}"
    );
    TrainedTopology::of(&input)
        .assert_valid(creature, "train_deadline test")
        .unwrap();
}

/// A stub scorer that counts its launches in `calls` and reports a fixed score.
fn write_stub_scorer(dir: &Path, calls: &Path) -> PathBuf {
    let path = dir.join("stub-scorer");
    let body = format!(
        r#"#!/bin/sh
set -eu
printf 'call\n' >> '{calls}'
printf '{{"trained":{{"score":1.0,"error":0.0}}}}\n'
"#,
        calls = calls.display(),
    );
    fs::write(&path, body).expect("write stub scorer");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod stub scorer");
    path
}

#[test]
fn expiry_during_the_initial_evaluation() {
    let data = corpus();
    let setup = Setup::default();
    let input = input_creature();

    let (_, run) = first_tick(
        "BaselineEvaluation",
        data.path(),
        &setup,
        timed_out_in(TimeoutPhase::BaselineEvaluation),
    );
    assert!(
        matches!(
            run.result.baseline,
            ErrorMeasure::Partial { .. } | ErrorMeasure::Unmeasured
        ),
        "baseline was {:?}",
        run.result.baseline
    );
    assert_eq!(
        status_of(&run),
        (TimeoutPhase::BaselineEvaluation, 0, false)
    );
    assert_eq!(parameters(&run.result.creature), parameters(&input));
    assert_eq!(run.result.accepted_epochs, 0);
    assert_eq!(run.result.best, run.result.baseline);
    assert!(epoch_lines(&run).is_empty(), "no epoch may be journalled");
    assert_eq!(run.journal.last().unwrap()["kind"], "timedOut");
    assert_valid_topology(&run.result.creature);
}

#[test]
fn expiry_on_the_very_first_tick_measures_nothing() {
    let data = corpus();
    let run = run_at_tick(data.path(), &Setup::default(), 1);

    assert_eq!(status_of(&run), (TimeoutPhase::Start, 0, false));
    assert_eq!(run.result.baseline, ErrorMeasure::Unmeasured);
    assert_eq!(run.result.baseline.mse(), None, "no fabricated mse");
    assert_eq!(run.result.best, ErrorMeasure::Unmeasured);
    assert_eq!(
        parameters(&run.result.creature),
        parameters(&input_creature())
    );
    assert_eq!(run.journal.last().unwrap()["kind"], "timedOut");
    assert_valid_topology(&run.result.creature);
}

#[test]
fn expiry_during_accumulation_in_the_first_epoch_keeps_useful_partial_learning() {
    let data = corpus();
    let setup = Setup::default();
    let input = input_creature();

    let (_, run) = first_tick("Accumulation (partial epoch)", data.path(), &setup, |run| {
        matches!(
            run.result.status,
            TrainStatus::TimedOut {
                phase: TimeoutPhase::Accumulation,
                completed_epochs: 0,
                weights_changed: true,
            }
        ) && epoch_lines(run)
            .iter()
            .any(|l| l["partial"] == true && l["accepted"] == true)
    });

    let epochs = epoch_lines(&run);
    assert_eq!(epochs.len(), 1, "one partial epoch line");
    assert_eq!(epochs[0]["partial"], true);
    let records = epochs[0]["records"].as_u64().unwrap();
    assert!(
        records >= 1 && records < u64::from(RECORDS),
        "partial epoch covers {records} records"
    );
    assert_eq!(run.journal.last().unwrap()["kind"], "timedOut");

    assert!(run.result.baseline.is_full(), "baseline must stay Full");
    assert!(
        matches!(run.result.best, ErrorMeasure::Partial { .. }),
        "best was {:?}",
        run.result.best
    );
    assert_eq!(run.result.accepted_epochs, 1);
    assert_ne!(parameters(&run.result.creature), parameters(&input));
    assert!(
        parameters(&run.result.creature)
            .iter()
            .all(|v| v.is_finite())
    );

    let before = corpus_mse(&input, data.path());
    let after = corpus_mse(&run.result.creature, data.path());
    assert!(
        after < before,
        "corpus MSE should fall: input {before}, returned {after}"
    );
    assert_valid_topology(&run.result.creature);
}

#[test]
fn accumulation_expiring_before_any_record_journals_nothing() {
    let data = corpus();
    let setup = Setup::default();

    let (_, run) = first_tick(
        "Accumulation",
        data.path(),
        &setup,
        timed_out_in(TimeoutPhase::Accumulation),
    );
    assert_eq!(status_of(&run), (TimeoutPhase::Accumulation, 0, false));
    assert!(epoch_lines(&run).is_empty(), "no epoch line");
    assert_eq!(
        parameters(&run.result.creature),
        parameters(&input_creature())
    );
    assert_eq!(run.result.accepted_epochs, 0);
    assert_eq!(run.result.best, run.result.baseline);
    assert_valid_topology(&run.result.creature);
}

#[test]
fn expiry_during_candidate_evaluation() {
    let data = corpus();
    let setup = Setup::default();
    let input = input_creature();

    // Expiry before the first candidate batch: nothing measured, nothing kept.
    let (_, run) = first_tick(
        "CandidateEvaluation",
        data.path(),
        &setup,
        timed_out_in(TimeoutPhase::CandidateEvaluation),
    );
    assert_eq!(
        status_of(&run),
        (TimeoutPhase::CandidateEvaluation, 0, false)
    );
    assert!(epoch_lines(&run).is_empty(), "no epoch line");
    assert_eq!(parameters(&run.result.creature), parameters(&input));
    assert!(run.result.baseline.is_full());
    assert_valid_topology(&run.result.creature);

    // Expiry after at least one candidate batch: a verified partial epoch.
    let (_, run) = first_tick(
        "CandidateEvaluation (verified partial epoch)",
        data.path(),
        &setup,
        |run| {
            timed_out_in(TimeoutPhase::CandidateEvaluation)(run)
                && epoch_lines(run)
                    .iter()
                    .any(|l| l["partial"] == true && l["accepted"] == true)
        },
    );
    let epoch = epoch_lines(&run)[0].clone();
    assert_eq!(epoch["partial"], true);
    assert!(
        epoch["beforeMse"].as_f64().unwrap() > epoch["afterMse"].as_f64().unwrap(),
        "an accepted partial candidate must verify better: {epoch}"
    );
    assert_ne!(parameters(&run.result.creature), parameters(&input));
    assert!(
        matches!(run.result.best, ErrorMeasure::Partial { .. }),
        "a changed run must not claim a full rescore: {:?}",
        run.result.best
    );
    assert_valid_topology(&run.result.creature);
}

#[test]
fn a_later_epoch_interruption_retains_earlier_accepted_learning() {
    let data = corpus();
    let setup = Setup {
        epochs: 5,
        ..Setup::default()
    };
    let input = input_creature();

    let (_, run) = first_tick(
        "Accumulation after a completed epoch",
        data.path(),
        &setup,
        |run| {
            matches!(
                run.result.status,
                TrainStatus::TimedOut {
                    phase: TimeoutPhase::Accumulation,
                    completed_epochs,
                    ..
                } if completed_epochs >= 1
            )
        },
    );
    assert!(run.result.accepted_epochs >= 1, "earlier learning kept");
    assert_ne!(parameters(&run.result.creature), parameters(&input));
    let complete: Vec<&Value> = epoch_lines(&run)
        .into_iter()
        .filter(|l| l["partial"] == false)
        .collect();
    let (_, completed, _) = status_of(&run);
    assert_eq!(
        complete.len() as u64,
        completed,
        "earlier epoch lines present"
    );
    assert_valid_topology(&run.result.creature);
}

#[test]
fn expiry_between_backtracking_attempts_stops_the_retry_loop() {
    let data = corpus();
    // A step this large overshoots, so attempt 0 is rejected. The precondition
    // is checked against an unbounded run below rather than assumed.
    let setup = Setup {
        learning_rate: 10.0,
        apply: ApplyOptions {
            step_scale: 1.0,
            ..ApplyOptions::default()
        },
        max_backtracks: 6,
        ..Setup::default()
    };
    let rejected_first = run_until(
        data.path(),
        &Setup {
            max_backtracks: 0,
            ..setup.clone()
        },
        None,
    );
    assert_eq!(
        rejected_first.result.accepted_epochs, 0,
        "precondition: attempt 0 must be rejected"
    );

    let (_, run) = first_tick(
        "Backtracking",
        data.path(),
        &setup,
        timed_out_in(TimeoutPhase::Backtracking),
    );
    assert_eq!(status_of(&run), (TimeoutPhase::Backtracking, 1, false));
    assert_eq!(run.result.accepted_epochs, 0);
    assert_eq!(
        parameters(&run.result.creature),
        parameters(&input_creature())
    );
    let epochs = epoch_lines(&run);
    assert_eq!(epochs.len(), 1, "the rejected epoch is journalled");
    assert_eq!(epochs[0]["accepted"], false);
    assert_eq!(epochs[0]["backtracks"], 0);
    assert_eq!(epochs[0]["partial"], false);
    assert_eq!(run.journal.last().unwrap()["kind"], "timedOut");
    assert_valid_topology(&run.result.creature);
}

#[test]
fn finalisation_launches_no_scorer_after_expiry() {
    let data = corpus();
    let stub_dir = tempdir().unwrap();
    let calls = stub_dir.path().join("calls.txt");
    let stub = write_stub_scorer(stub_dir.path(), &calls);
    let setup = Setup {
        epochs: 0,
        scorer: Some(stub),
        ..Setup::default()
    };

    let (_, run) = first_tick(
        "Scoring",
        data.path(),
        &setup,
        timed_out_in(TimeoutPhase::Scoring),
    );
    assert_eq!(status_of(&run), (TimeoutPhase::Scoring, 0, false));
    assert!(
        fs::read_to_string(&calls).map_or(true, |s| s.is_empty()),
        "the scorer must not launch after expiry"
    );
    assert!(run.result.baseline_score.is_none());
    assert!(run.result.best_score.is_none());
    let best: Value =
        serde_json::from_str(&fs::read_to_string(run.out.path().join("best.json")).unwrap())
            .expect("best.json is valid JSON");
    assert!(best.is_object());
    assert_eq!(run.journal.last().unwrap()["kind"], "timedOut");
}

#[test]
fn an_already_expired_deadline_returns_the_input_unchanged_with_no_measurement() {
    let data = corpus();
    let deadline = Deadline::at_epoch_ms(1);
    let run = run_until(data.path(), &Setup::default(), Some(&deadline));

    assert_eq!(status_of(&run), (TimeoutPhase::Start, 0, false));
    assert_eq!(run.result.baseline, ErrorMeasure::Unmeasured);
    assert_eq!(run.result.best, ErrorMeasure::Unmeasured);
    assert_eq!(run.result.accepted_epochs, 0);
    let before: Vec<u64> = parameters(&input_creature())
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let after: Vec<u64> = parameters(&run.result.creature)
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(after, before, "weights are bit-equal to the input");
}

#[test]
fn a_distant_deadline_matches_the_unbounded_run() {
    let data = corpus();
    let setup = Setup {
        epochs: 3,
        ..Setup::default()
    };

    let deadline = Deadline::at_epoch_ms(u64::MAX);
    let bounded = run_until(data.path(), &setup, Some(&deadline));

    let out = tempdir().unwrap();
    let config = config_for(&setup);
    let unbounded = run_train(request(data.path(), out.path(), &config, &setup)).unwrap();
    let unbounded_journal = read_journal(out.path());

    assert_eq!(bounded.result.status, TrainStatus::Completed);
    assert_eq!(bounded.result.best_json, unbounded.best_json);
    assert_eq!(bounded.result.accepted_epochs, unbounded.accepted_epochs);
    assert_eq!(epoch_lines(&bounded).len(), {
        unbounded_journal
            .iter()
            .filter(|l| l["kind"] == "epoch")
            .count()
    });
    assert!(bounded.result.baseline.is_full());
    assert!(bounded.result.best.is_full());
    assert!(nearly_equal(
        bounded.result.best.full_mse().unwrap(),
        unbounded.best.full_mse().unwrap()
    ));
    assert!(nearly_equal(
        bounded.result.baseline.full_mse().unwrap(),
        unbounded.baseline.full_mse().unwrap()
    ));
    assert!(
        unbounded_journal.iter().all(|l| l["kind"] != "timedOut"),
        "a run that does not time out journals no timedOut line"
    );
    assert!(epoch_lines(&bounded).iter().all(|l| l["partial"] == false));
}

#[test]
fn an_epoch_line_without_the_partial_key_reads_as_complete() {
    let data = corpus();
    let run = run_until(data.path(), &Setup::default(), None);
    let mut line = epoch_lines(&run)[0].clone();
    assert_eq!(line["partial"], false, "new lines carry partial:false");

    line.as_object_mut().unwrap().remove("partial");
    let record: TrainEpochRecord = serde_json::from_value(line).unwrap();
    assert!(!record.partial, "an old journal line is a complete epoch");
}
