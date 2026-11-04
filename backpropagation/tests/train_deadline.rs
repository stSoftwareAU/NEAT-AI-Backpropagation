//! Issue #213: `run_train_until` stops cooperatively at an absolute deadline
//! inside an epoch, keeps the learning it already accepted, and returns useful
//! partial learning.
//!
//! Every phase is reached deterministically. A counting clock expires on its
//! `T`-th read, and [`first_tick`] scans `T` upward until a run stops in the
//! phase under test. Because the scan only lands on a tick where that phase's
//! own check exists, each test fails when only that check is removed.

#![cfg(unix)]

use neat_ai_backpropagation::backprop::{ApplyOptions, BackpropConfig};
use neat_ai_backpropagation::deadline::{Deadline, TimeoutPhase};
use neat_ai_backpropagation::mse::compute_mse;
use neat_ai_backpropagation::train::{
    AcceptanceMode, PARTIAL_VERIFY_RECORDS, ScorerAcceptance, TrainCreature, TrainEpochRecord,
    TrainRequest, TrainResult, TrainStatus, run_train, run_train_until,
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

/// Records per batch in a slice-MSE scan. Mirrors `DEADLINE_CHECK_RECORDS` in
/// `src/mse.rs`: the clock is read before each batch.
const BATCH_RECORDS: u64 = 32;

/// Clock reads one rung's slice-MSE scan makes over [`RECORDS`]: one before each
/// batch, so `ceil(100 / 32) = 4`.
const SCAN_CHECKS: u64 = (RECORDS as u64).div_ceil(BATCH_RECORDS);

/// Corpus of `(x, 2x)` pairs with `x` cycling through `0.1 ..= 1.0`. Every
/// input is positive, so one gradient direction improves the whole corpus.
fn corpus() -> TempDir {
    corpus_with_records(RECORDS)
}

fn corpus_with_records(records: u32) -> TempDir {
    let dir = tempdir().unwrap();
    let mut f = fs::File::create(dir.path().join("0.bin")).unwrap();
    for i in 0..records {
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
    trace_store: Option<PathBuf>,
    acceptance: AcceptanceMode,
    ladder: Vec<f64>,
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
            trace_store: None,
            acceptance: AcceptanceMode::Mse,
            ladder: Vec::new(),
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
        acceptance: setup.acceptance,
        accept_always: false,
        max_backtracks: setup.max_backtracks,
        step_scale_ladder: setup.ladder.as_slice(),
        trace_store: setup.trace_store.as_deref(),
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

/// A stub for `rust_scorer <creature dir> <training data>`, the boundary
/// contract in `backpropagation/src/scorer.rs` (`score_creature` writes the
/// candidate as `trained.json` into the directory it passes, and the training
/// data path is passed through unchanged) and `backpropagation/tests/scorer_boundary.rs`,
/// which fakes the same boundary.
///
/// The stub exits non-zero with a stderr message unless it gets exactly two
/// arguments, a creature directory holding `trained.json`, and a training-data
/// directory. Each launch is tallied in `calls`; a valid launch reports a score
/// of `1.0`.
fn write_stub_scorer(dir: &Path, calls: &Path) -> PathBuf {
    let path = dir.join("stub-scorer");
    let body = format!(
        r#"#!/bin/sh
set -eu
printf 'call\n' >> '{calls}'
if [ "$#" -ne 2 ]; then
  echo "stub scorer: expected <creature dir> <training data>, got $# argument(s)" >&2
  exit 2
fi
if [ ! -d "$1" ] || [ ! -f "$1/trained.json" ]; then
  echo "stub scorer: $1 is not a creature directory holding trained.json" >&2
  exit 3
fi
if [ ! -d "$2" ]; then
  echo "stub scorer: training data $2 is not a directory" >&2
  exit 4
fi
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
fn post_deadline_verification_is_capped_for_large_corpus() {
    const LARGE_CORPUS: u32 = 300;
    assert!(PARTIAL_VERIFY_RECORDS < u64::from(LARGE_CORPUS));
    let data = corpus_with_records(LARGE_CORPUS);
    let setup = Setup::default();
    let (accumulation_start, _) = first_tick(
        "start of accumulation",
        data.path(),
        &setup,
        timed_out_in(TimeoutPhase::Accumulation),
    );
    let (candidate_tick, candidate) = first_tick(
        "first candidate batch",
        data.path(),
        &setup,
        timed_out_in(TimeoutPhase::CandidateEvaluation),
    );
    // Two reads earlier is in accumulation, after more than the cap.
    assert!(candidate_tick - 2 - accumulation_start > PARTIAL_VERIFY_RECORDS);
    let accumulation = run_at_tick(data.path(), &setup, candidate_tick - 2);
    assert_eq!(status_of(&accumulation).0, TimeoutPhase::Accumulation);
    for run in [&candidate, &accumulation] {
        let epochs = epoch_lines(run);
        assert_eq!(epochs.len(), 1);
        assert_eq!(epochs[0]["partial"], true);
        assert_eq!(epochs[0]["accepted"], true);
        assert_eq!(epochs[0]["records"].as_u64(), Some(PARTIAL_VERIFY_RECORDS));
        assert!(
            matches!(run.result.best, ErrorMeasure::Partial { records, .. } if records == PARTIAL_VERIFY_RECORDS),
            "best was {:?}",
            run.result.best
        );
    }
}

#[test]
fn expiry_during_candidate_evaluation() {
    let data = corpus();
    let setup = Setup::default();
    let input = input_creature();

    // Expiry before the first candidate batch: nothing of the candidate was
    // measured, but the accumulation completed, so plain MSE still verifies the
    // single candidate over the records it covers (min(100, 256) = 100) and keeps
    // or rejects its learning on that verdict (#213).
    let (_, run) = first_tick(
        "CandidateEvaluation",
        data.path(),
        &setup,
        timed_out_in(TimeoutPhase::CandidateEvaluation),
    );
    let epochs = epoch_lines(&run);
    assert_eq!(epochs.len(), 1, "one partial epoch line");
    let epoch = epochs[0];
    assert_eq!(epoch["partial"], true, "{epoch}");
    assert_eq!(
        epoch["records"].as_u64(),
        Some(u64::from(RECORDS).min(PARTIAL_VERIFY_RECORDS)),
        "verification window: {epoch}"
    );
    let accepted = epoch["accepted"] == true;
    let before = epoch["beforeMse"].as_f64().unwrap();
    let after = epoch["afterMse"].as_f64().unwrap();
    assert_eq!(
        status_of(&run),
        (TimeoutPhase::CandidateEvaluation, 0, accepted)
    );
    assert_eq!(run.result.accepted_epochs, u64::from(accepted));
    assert!(run.result.baseline.is_full());
    if accepted {
        assert!(after < before, "accepted must verify better: {epoch}");
        assert_ne!(parameters(&run.result.creature), parameters(&input));
        assert!(
            matches!(run.result.best, ErrorMeasure::Partial { records, .. } if records == 100),
            "best was {:?}",
            run.result.best
        );
    } else {
        assert!(after >= before, "rejected must not verify better: {epoch}");
        assert_eq!(parameters(&run.result.creature), parameters(&input));
        assert_eq!(run.result.best, run.result.baseline);
    }
    assert_eq!(run.journal.last().unwrap()["kind"], "timedOut");
    assert_valid_topology(&run.result.creature);

    // Expiry after at least one candidate batch: a verified partial epoch over
    // the records measured so far, which is fewer than the corpus.
    let (_, run) = first_tick(
        "CandidateEvaluation (verified partial epoch)",
        data.path(),
        &setup,
        |run| {
            timed_out_in(TimeoutPhase::CandidateEvaluation)(run)
                && epoch_lines(run).iter().any(|l| {
                    l["partial"] == true
                        && l["accepted"] == true
                        && l["records"]
                            .as_u64()
                            .is_some_and(|r| r < u64::from(RECORDS))
                })
        },
    );
    let epoch = epoch_lines(&run)[0].clone();
    assert_eq!(epoch["partial"], true);
    let records = epoch["records"].as_u64().unwrap();
    let before = epoch["beforeMse"].as_f64().unwrap();
    let mut network = compile_creature(&input).unwrap();
    let (window_mse, measured) =
        compute_mse(&input, &mut network, data.path(), Some(records)).unwrap();
    assert_eq!(measured, records);
    assert!(
        nearly_equal(before, window_mse),
        "beforeMse must cover the verification window: {epoch}"
    );
    assert!(
        !nearly_equal(before, run.result.baseline.full_mse().unwrap()),
        "fixture must distinguish the window from the full corpus: {epoch}"
    );
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

/// Expiry at the top of a later epoch, before its accumulation starts, is
/// reported as `EpochStart`. The completed epochs stay, and no partial line is
/// journalled for the epoch that never began.
#[test]
fn expiry_at_the_top_of_a_later_epoch_is_reported_as_epoch_start() {
    let data = corpus();
    let setup = Setup {
        epochs: 5,
        ..Setup::default()
    };
    let input = input_creature();

    let (tick, run) = first_tick("EpochStart", data.path(), &setup, |run| {
        matches!(
            run.result.status,
            TrainStatus::TimedOut {
                phase: TimeoutPhase::EpochStart,
                weights_changed: true,
                ..
            }
        )
    });
    let (phase, completed, weights_changed) = status_of(&run);
    assert_eq!(
        (phase, weights_changed),
        (TimeoutPhase::EpochStart, true),
        "tick {tick}"
    );
    assert_eq!(completed, 1);
    let epochs = epoch_lines(&run);
    assert_eq!(
        epochs.len() as u64,
        completed,
        "one line per completed epoch"
    );
    assert!(
        epochs.iter().all(|l| l["partial"] == false),
        "no partial epoch after the stop"
    );
    assert!(run.result.accepted_epochs >= 1);
    assert_ne!(parameters(&run.result.creature), parameters(&input));
    assert_eq!(run.journal.last().unwrap()["kind"], "timedOut");
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

/// Plain MSE acceptance with a scorer: expiry after the baseline finalisation
/// scan has launched must not launch the best creature's scan (#213). With no
/// epochs the only scorer launch is the baseline one, so the expiry lands between
/// it and the best-score scan.
#[test]
fn finalisation_does_not_launch_the_best_scorer_after_expiry() {
    let data = corpus();
    let stub_dir = tempdir().unwrap();
    let calls = stub_dir.path().join("calls.txt");
    let stub = write_stub_scorer(stub_dir.path(), &calls);
    let setup = Setup {
        epochs: 0,
        scorer: Some(stub),
        ..Setup::default()
    };

    let (tick, run, _) = first_scored_tick(
        "Scoring before the best-creature scan",
        data.path(),
        &setup,
        &calls,
        |_, run, count| count == 1 && timed_out_in(TimeoutPhase::Scoring)(run),
    );
    assert_eq!(
        status_of(&run),
        (TimeoutPhase::Scoring, 0, false),
        "tick {tick}"
    );
    assert_eq!(launches(&calls), 1, "only the baseline scan launched");
    assert!(
        run.result.baseline_score.is_some(),
        "the baseline was scored"
    );
    assert!(run.result.best_score.is_none(), "no best-creature score");
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

/// Three rungs. Each rung reads the clock on entry and once before each of its
/// [`SCAN_CHECKS`] record batches; the batch scorer's check is read straight
/// after the last rung's scan.
const LADDER: [f64; 3] = [0.001, 0.005, 0.01];

/// Scorer-guided acceptance with the default epsilon and no MSE pre-screen.
fn scorer_guided() -> AcceptanceMode {
    AcceptanceMode::Scorer(ScorerAcceptance::default())
}

/// A batch stand-in for `rust_scorer <creature dir> <training data>`, the
/// contract in `backpropagation/src/scorer.rs` (`score_creatures` writes one
/// `<stem>.json` per candidate into the directory it passes, and the training
/// data path is passed through unchanged) and `backpropagation/tests/scorer_boundary.rs`.
///
/// It scores every creature file in the directory, one `"stem"` entry each, as
/// the real scorer does. It exits non-zero with a stderr message unless it gets
/// exactly two arguments, a creature directory holding at least one `.json`
/// file, and a training-data directory. Each launch is tallied in `calls`; every
/// score is `1.0`.
fn write_batch_stub_scorer(dir: &Path, calls: &Path) -> PathBuf {
    let path = dir.join("batch-stub-scorer");
    let body = format!(
        r#"#!/bin/sh
set -eu
printf 'call\n' >> '{calls}'
if [ "$#" -ne 2 ]; then
  echo "batch stub scorer: expected <creature dir> <training data>, got $# argument(s)" >&2
  exit 2
fi
if [ ! -d "$1" ]; then
  echo "batch stub scorer: $1 is not a creature directory" >&2
  exit 3
fi
if [ ! -d "$2" ]; then
  echo "batch stub scorer: training data $2 is not a directory" >&2
  exit 4
fi
out=''
for candidate in "$1"/*.json; do
  [ -e "$candidate" ] || continue
  stem=$(basename "$candidate" .json)
  if [ -n "$out" ]; then
    out="$out,"
  fi
  out="$out\"$stem\":{{\"score\":1.0,\"error\":0.0}}"
done
if [ -z "$out" ]; then
  echo "batch stub scorer: $1 holds no creature .json files" >&2
  exit 5
fi
printf '{{%s}}\n' "$out"
"#,
        calls = calls.display(),
    );
    fs::write(&path, body).expect("write batch stub scorer");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod batch stub scorer");
    path
}

/// A scorer-guided setup whose scorer tallies its launches in the returned
/// `calls` file. A non-empty `ladder` gets the batch stub, which the ladder's
/// single batch call needs; otherwise the single-creature stub is used.
fn scorer_setup(stub_dir: &Path, ladder: &[f64]) -> (Setup, PathBuf) {
    let calls = stub_dir.join("calls.txt");
    let scorer = if ladder.is_empty() {
        write_stub_scorer(stub_dir, &calls)
    } else {
        write_batch_stub_scorer(stub_dir, &calls)
    };
    let setup = Setup {
        acceptance: scorer_guided(),
        scorer: Some(scorer),
        ladder: ladder.to_vec(),
        ..Setup::default()
    };
    (setup, calls)
}

/// Scorer launches tallied in `calls`. A missing tally is zero launches.
fn launches(calls: &Path) -> usize {
    match fs::read_to_string(calls) {
        Ok(text) => text.lines().count(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => panic!("cannot read {}: {e}", calls.display()),
    }
}

/// [`first_tick`] for the scorer tests. Each run starts with an empty tally, so
/// `wanted` sees exactly that run's launches, and it also receives the tick.
/// Panics with a clear message when no tick up to [`MAX_TICK`] matches, which
/// is what a removed check causes.
fn first_scored_tick(
    phase: &str,
    data: &Path,
    setup: &Setup,
    calls: &Path,
    wanted: impl Fn(u64, &Run, usize) -> bool,
) -> (u64, Run, usize) {
    for tick in 1..=MAX_TICK {
        let _ = fs::remove_file(calls);
        let run = run_at_tick(data, setup, tick);
        let count = launches(calls);
        if wanted(tick, &run, count) {
            return (tick, run, count);
        }
    }
    panic!("no expiry tick reaches phase {phase}");
}

/// The tick of the ladder's first clock read: rung 0's entry check, which
/// precedes its apply. It is the first tick at which a ladder run stops in
/// `CandidateEvaluation` with only the baseline launched.
fn ladder_entry_tick(data: &Path, setup: &Setup, calls: &Path) -> u64 {
    first_scored_tick(
        "CandidateEvaluation at rung 0's entry",
        data,
        setup,
        calls,
        |_, run, count| count == 1 && timed_out_in(TimeoutPhase::CandidateEvaluation)(run),
    )
    .0
}

/// Scorer-guided acceptance: a run that expires before the baseline scorer
/// launches must not launch it, and must score and judge nothing.
#[test]
fn scorer_baseline_is_not_launched_after_expiry() {
    let data = corpus();
    let stub_dir = tempdir().unwrap();
    let (setup, calls) = scorer_setup(stub_dir.path(), &[]);
    let input = input_creature();

    let (tick, run, _) = first_scored_tick(
        "Scoring before the baseline launch",
        data.path(),
        &setup,
        &calls,
        |_, run, count| count == 0 && timed_out_in(TimeoutPhase::Scoring)(run),
    );
    assert_eq!(
        status_of(&run),
        (TimeoutPhase::Scoring, 0, false),
        "tick {tick}"
    );
    assert!(run.result.baseline.is_full(), "the baseline pass finished");
    assert!(run.result.baseline_score.is_none());
    assert!(run.result.best_score.is_none());
    assert_eq!(run.result.accepted_epochs, 0);
    assert_eq!(parameters(&run.result.creature), parameters(&input));
    assert!(epoch_lines(&run).is_empty(), "no epoch line");
    assert_eq!(run.journal.last().unwrap()["kind"], "timedOut");
    assert_valid_topology(&run.result.creature);
}

/// Scorer-guided acceptance: a run that expires after the baseline is scored
/// but before the candidate launches must not score the candidate, and must
/// keep the incumbent.
#[test]
fn scorer_candidate_is_not_launched_after_expiry() {
    let data = corpus();
    let stub_dir = tempdir().unwrap();
    let (setup, calls) = scorer_setup(stub_dir.path(), &[]);
    let input = input_creature();

    let (tick, run, _) = first_scored_tick(
        "Scoring before the candidate launch",
        data.path(),
        &setup,
        &calls,
        |_, run, count| count == 1 && timed_out_in(TimeoutPhase::Scoring)(run),
    );
    assert_eq!(
        status_of(&run),
        (TimeoutPhase::Scoring, 0, false),
        "tick {tick}"
    );
    assert!(
        run.result.baseline_score.is_some(),
        "the baseline was scored"
    );
    assert_eq!(run.result.best, run.result.baseline);
    assert_eq!(run.result.accepted_epochs, 0);
    assert_eq!(parameters(&run.result.creature), parameters(&input));
    assert!(epoch_lines(&run).is_empty(), "no epoch line");
    assert!(
        run.journal.iter().all(|l| l["kind"] != "candidate"),
        "a candidate that never launched was judged"
    );
    assert_valid_topology(&run.result.creature);
}

/// Expiry between two rungs of a ladder stops the epoch there: no rung is
/// scored, the incumbent stays, and nothing is journalled for the epoch.
#[test]
fn a_ladder_run_stops_between_rungs() {
    let data = corpus();
    let stub_dir = tempdir().unwrap();
    let (setup, calls) = scorer_setup(stub_dir.path(), &LADDER);
    let input = input_creature();

    let entry = ladder_entry_tick(data.path(), &setup, &calls);
    // Rung 0's scan reads the clock SCAN_CHECKS times, so the first tick past it
    // is rung 1's entry check: the first stop between rungs.
    let (tick, run, _) = first_scored_tick(
        "CandidateEvaluation between rungs",
        data.path(),
        &setup,
        &calls,
        |tick, run, count| {
            tick > entry + SCAN_CHECKS
                && count == 1
                && timed_out_in(TimeoutPhase::CandidateEvaluation)(run)
        },
    );
    assert_eq!(tick, entry + SCAN_CHECKS + 1, "rung 1's entry check");
    assert_eq!(
        status_of(&run),
        (TimeoutPhase::CandidateEvaluation, 0, false),
        "tick {tick}"
    );
    assert!(
        run.result.baseline_score.is_some(),
        "the baseline was scored"
    );
    assert_eq!(run.result.best, run.result.baseline);
    assert_eq!(run.result.accepted_epochs, 0);
    assert_eq!(parameters(&run.result.creature), parameters(&input));
    assert!(epoch_lines(&run).is_empty(), "no epoch line");
    assert!(
        run.journal.iter().all(|l| l["kind"] != "candidate"),
        "no ladder rung may be journalled"
    );
    assert_valid_topology(&run.result.creature);
}

/// Expiry inside rung 0's slice-MSE scan stops the ladder there (#213). The last
/// in-scan batch check of the first rung is a `CandidateEvaluation` stop with
/// only the baseline launched, and no rung is judged or journalled. Without the
/// in-scan check the scan runs to completion, so no tick at or after this one
/// stops before the batch launch, and the scan finds none.
#[test]
fn a_ladder_rung_is_interrupted_inside_its_scan() {
    let data = corpus();
    let stub_dir = tempdir().unwrap();
    let (setup, calls) = scorer_setup(stub_dir.path(), &LADDER);
    let input = input_creature();

    let entry = ladder_entry_tick(data.path(), &setup, &calls);
    let (tick, run, _) = first_scored_tick(
        "CandidateEvaluation inside rung 0's scan",
        data.path(),
        &setup,
        &calls,
        |tick, run, count| {
            tick >= entry + SCAN_CHECKS
                && count == 1
                && timed_out_in(TimeoutPhase::CandidateEvaluation)(run)
        },
    );
    assert_eq!(
        tick,
        entry + SCAN_CHECKS,
        "the last batch check of rung 0's scan"
    );
    assert_eq!(
        status_of(&run),
        (TimeoutPhase::CandidateEvaluation, 0, false),
        "tick {tick}"
    );
    assert_eq!(launches(&calls), 1, "only the baseline launched");
    assert_eq!(run.result.accepted_epochs, 0);
    assert_eq!(parameters(&run.result.creature), parameters(&input));
    assert!(epoch_lines(&run).is_empty(), "no epoch line");
    assert!(
        run.journal.iter().all(|l| l["kind"] != "candidate"),
        "no rung may be journalled"
    );
    assert_valid_topology(&run.result.creature);
}

/// Expiry at the ladder's batch check launches no `rust_scorer` batch, even
/// though every rung was applied and measured before it.
#[test]
fn a_ladder_run_does_not_launch_its_batch_scorer_after_expiry() {
    let data = corpus();
    let stub_dir = tempdir().unwrap();
    let (setup, calls) = scorer_setup(stub_dir.path(), &LADDER);
    let input = input_creature();

    let entry = ladder_entry_tick(data.path(), &setup, &calls);
    let (batch, run, _) = first_scored_tick(
        "Scoring before the ladder batch",
        data.path(),
        &setup,
        &calls,
        |_, run, count| count == 1 && timed_out_in(TimeoutPhase::Scoring)(run),
    );
    assert_eq!(
        status_of(&run),
        (TimeoutPhase::Scoring, 0, false),
        "tick {batch}"
    );
    assert_eq!(run.result.accepted_epochs, 0);
    assert_eq!(parameters(&run.result.creature), parameters(&input));
    assert!(epoch_lines(&run).is_empty(), "no epoch line");
    assert!(
        run.journal.iter().all(|l| l["kind"] != "candidate"),
        "no rung may be journalled"
    );
    assert_valid_topology(&run.result.creature);

    // Every ladder read, the rung entries and the in-scan batch checks alike,
    // stops with only the baseline launched. The batch check is the read straight
    // after the last rung's scan.
    let ladder_reads = LADDER.len() as u64 * (1 + SCAN_CHECKS);
    assert_eq!(
        batch,
        entry + ladder_reads,
        "the batch check follows the last rung's scan"
    );
    let between: Vec<u64> = (entry..batch)
        .filter(|&tick| {
            let _ = fs::remove_file(&calls);
            let stopped = run_at_tick(data.path(), &setup, tick);
            launches(&calls) == 1 && timed_out_in(TimeoutPhase::CandidateEvaluation)(&stopped)
        })
        .collect();
    assert_eq!(
        between.len() as u64,
        ladder_reads,
        "ladder ticks {between:?} before the batch tick {batch}"
    );
}

/// A scorer-guided or ladder run cut inside the epoch has no scorer verdict for
/// its partial candidate, so it must keep the incumbent: no epoch or candidate
/// line, nothing accepted, parameters unchanged, `weights_changed` false.
fn assert_incumbent_kept(run: &Run, phase: TimeoutPhase, tick: u64) {
    assert_eq!(status_of(run), (phase, 0, false), "tick {tick}");
    assert_eq!(run.result.accepted_epochs, 0, "tick {tick}");
    assert_eq!(
        parameters(&run.result.creature),
        parameters(&input_creature()),
        "tick {tick}"
    );
    assert!(epoch_lines(run).is_empty(), "no epoch line at tick {tick}");
    assert!(
        run.journal.iter().all(|l| l["kind"] != "candidate"),
        "no candidate line at tick {tick}"
    );
    assert_valid_topology(&run.result.creature);
}

/// Positive control for the scorer tests below: plain-MSE acceptance at `tick`
/// writes a partial epoch line, so the same stop in a scorer-guided or ladder run
/// reaches the partial path that the guard under test refuses.
fn assert_mse_run_judges_a_partial_epoch(data: &Path, tick: u64) {
    let run = run_at_tick(data, &Setup::default(), tick);
    let epochs = epoch_lines(&run);
    assert_eq!(
        epochs.len(),
        1,
        "plain MSE judges a partial epoch at {tick}"
    );
    assert_eq!(epochs[0]["partial"], true);
}

/// The first accumulation stop with a record already accumulated, for a
/// scorer-guided run whose baseline was launched: one tick past the stop before
/// any record. Returns the tick and its plain-MSE counterpart, which has no
/// baseline scorer check so reads the clock one tick earlier.
fn scorer_accumulation_stop(data: &Path, setup: &Setup, calls: &Path) -> (u64, u64) {
    let (first, _, _) = first_scored_tick(
        "Accumulation before any record",
        data,
        setup,
        calls,
        |_, run, count| count == 1 && timed_out_in(TimeoutPhase::Accumulation)(run),
    );
    let (mse_first, _) = first_tick(
        "Accumulation (plain MSE)",
        data,
        &Setup::default(),
        timed_out_in(TimeoutPhase::Accumulation),
    );
    assert_eq!(first, mse_first + 1, "one baseline scorer check earlier");
    (first + 1, mse_first + 1)
}

/// Expiry mid-accumulation in a scorer-guided run keeps the incumbent: the
/// partial learning has no scorer verdict, so plain MSE must not accept it.
#[test]
fn a_scorer_guided_run_cut_mid_accumulation_keeps_the_incumbent() {
    let data = corpus();
    let stub_dir = tempdir().unwrap();
    let (setup, calls) = scorer_setup(stub_dir.path(), &[]);

    let (tick, mse_tick) = scorer_accumulation_stop(data.path(), &setup, &calls);
    assert_mse_run_judges_a_partial_epoch(data.path(), mse_tick);
    let _ = fs::remove_file(&calls);
    let run = run_at_tick(data.path(), &setup, tick);
    assert_eq!(launches(&calls), 1, "only the baseline launched");
    assert_incumbent_kept(&run, TimeoutPhase::Accumulation, tick);
}

/// The same stop with a step-scale ladder configured: the ladder half of the
/// partial-accumulation guard.
#[test]
fn a_ladder_run_cut_mid_accumulation_keeps_the_incumbent() {
    let data = corpus();
    let stub_dir = tempdir().unwrap();
    let (setup, calls) = scorer_setup(stub_dir.path(), &LADDER);

    let (tick, mse_tick) = scorer_accumulation_stop(data.path(), &setup, &calls);
    assert_mse_run_judges_a_partial_epoch(data.path(), mse_tick);
    let _ = fs::remove_file(&calls);
    let run = run_at_tick(data.path(), &setup, tick);
    assert_eq!(launches(&calls), 1, "only the baseline launched");
    assert_incumbent_kept(&run, TimeoutPhase::Accumulation, tick);
}

/// Expiry inside the scorer-guided candidate scan keeps the incumbent, both at
/// the first candidate batch (nothing measured) and after a later one (partly
/// measured). Only the baseline may have launched.
#[test]
fn a_scorer_guided_run_cut_mid_candidate_scan_keeps_the_incumbent() {
    let data = corpus();
    let stub_dir = tempdir().unwrap();
    let (setup, calls) = scorer_setup(stub_dir.path(), &[]);

    let (first, _, _) = first_scored_tick(
        "CandidateEvaluation at the first candidate batch",
        data.path(),
        &setup,
        &calls,
        |_, run, count| count == 1 && timed_out_in(TimeoutPhase::CandidateEvaluation)(run),
    );
    let (mse_first, _) = first_tick(
        "CandidateEvaluation (plain MSE)",
        data.path(),
        &Setup::default(),
        timed_out_in(TimeoutPhase::CandidateEvaluation),
    );
    assert_eq!(first, mse_first + 1, "one baseline scorer check earlier");

    // `first` measures nothing of the candidate; the later stops follow at least
    // one scanned batch, so the candidate is only partly measured.
    for (offset, mse_offset) in [(0, 0), (1, 1), (SCAN_CHECKS - 1, SCAN_CHECKS - 1)] {
        assert_mse_run_judges_a_partial_epoch(data.path(), mse_first + mse_offset);
        let tick = first + offset;
        let _ = fs::remove_file(&calls);
        let run = run_at_tick(data.path(), &setup, tick);
        assert_eq!(launches(&calls), 1, "only the baseline launched at {tick}");
        assert_incumbent_kept(&run, TimeoutPhase::CandidateEvaluation, tick);
    }
}

/// The partial-accumulation guard applies only to a partial epoch: a scorer-guided
/// or ladder run whose accumulation completes must still run its scorer path and
/// journal a complete (non-partial) epoch.
#[test]
fn a_scorer_guided_run_that_completes_accumulation_is_not_cut_short() {
    let data = corpus();
    let stub_dir = tempdir().unwrap();
    for ladder in [&[][..], &LADDER[..]] {
        let (setup, calls) = scorer_setup(stub_dir.path(), ladder);
        let _ = fs::remove_file(&calls);
        let run = run_until(data.path(), &setup, None);
        assert!(
            matches!(run.result.status, TrainStatus::Completed),
            "ladder {ladder:?}: {:?}",
            run.result.status
        );
        assert!(launches(&calls) >= 2, "baseline and candidate were scored");
        let epochs = epoch_lines(&run);
        assert_eq!(epochs.len(), 1, "one complete epoch line");
        assert_ne!(epochs[0]["partial"], true, "{}", epochs[0]);
    }
}

/// Corpus of `(x, x)` pairs: the identity chain already fits it exactly, so
/// every gradient is zero and any candidate ties the incumbent.
fn perfect_corpus() -> TempDir {
    let dir = tempdir().unwrap();
    let mut f = fs::File::create(dir.path().join("0.bin")).unwrap();
    for i in 0..RECORDS {
        let x = (i % 10 + 1) as f32 * 0.1;
        f.write_all(&x.to_le_bytes()).unwrap();
        f.write_all(&x.to_le_bytes()).unwrap();
    }
    dir
}

#[test]
fn a_partial_epoch_that_only_ties_the_incumbent_is_rejected() {
    let data = perfect_corpus();
    let setup = Setup::default();

    let (_, run) = first_tick(
        "Accumulation (tied partial epoch)",
        data.path(),
        &setup,
        |run| timed_out_in(TimeoutPhase::Accumulation)(run) && !epoch_lines(run).is_empty(),
    );

    let epochs = epoch_lines(&run);
    assert_eq!(epochs.len(), 1, "one partial epoch line");
    assert_eq!(epochs[0]["partial"], true);
    assert_eq!(
        epochs[0]["accepted"], false,
        "a tie is not an improvement under plain MSE"
    );
    assert_eq!(run.result.accepted_epochs, 0);
    assert_eq!(
        parameters(&run.result.creature),
        parameters(&input_creature())
    );
}

#[test]
fn partial_trace_classifies_improvement_and_ties() {
    let traces = tempdir().unwrap();
    let setup = Setup {
        trace_store: Some(traces.path().to_path_buf()),
        ..Setup::default()
    };
    let improving = corpus();
    let (_, run) = first_tick("improving partial", improving.path(), &setup, |run| {
        timed_out_in(TimeoutPhase::Accumulation)(run) && !epoch_lines(run).is_empty()
    });
    assert!(run.out.path().join("best-trace.json").is_file());
    assert!(!traces.path().join("failed/epoch-1.json").exists());

    let tied = perfect_corpus();
    let (_, run) = first_tick("tied partial", tied.path(), &setup, |run| {
        timed_out_in(TimeoutPhase::Accumulation)(run) && !epoch_lines(run).is_empty()
    });
    assert!(!run.out.path().join("best-trace.json").exists());
    assert!(traces.path().join("failed/epoch-1.json").is_file());
}

#[test]
fn partial_trace_compares_against_the_verified_window_not_the_full_corpus() {
    let data = tempdir().unwrap();
    let mut file = fs::File::create(data.path().join("0.bin")).unwrap();
    for i in 0..RECORDS {
        let x = 1.0_f32;
        let target = if i == 0 { 1.1_f32 } else { 10.0_f32 };
        file.write_all(&x.to_le_bytes()).unwrap();
        file.write_all(&target.to_le_bytes()).unwrap();
    }
    let traces = tempdir().unwrap();
    let setup = Setup {
        learning_rate: 10.0,
        apply: ApplyOptions {
            step_scale: 1.0,
            ..ApplyOptions::default()
        },
        trace_store: Some(traces.path().to_path_buf()),
        ..Setup::default()
    };
    let (_, run) = first_tick("partial trace baseline", data.path(), &setup, |run| {
        timed_out_in(TimeoutPhase::Accumulation)(run) && !epoch_lines(run).is_empty()
    });
    let epochs = epoch_lines(&run);
    let epoch = epochs[0];
    let before = epoch["beforeMse"].as_f64().unwrap();
    let after = epoch["afterMse"].as_f64().unwrap();
    assert_eq!(epoch["accepted"], false);
    assert!(
        after > before,
        "candidate must be worse on the verification window"
    );
    assert!(
        after < run.result.baseline.full_mse().unwrap(),
        "candidate must still beat the full-corpus baseline"
    );
    assert!(traces.path().join("failed/epoch-1.json").is_file());
    assert!(!run.out.path().join("best-trace.json").exists());
}
