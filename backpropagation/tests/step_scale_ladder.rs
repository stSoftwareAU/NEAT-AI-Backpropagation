//! Scored step-scale ladder: keep the best scorer winner (issue #106).
//!
//! The backtracking line search halves the step until the judge is satisfied
//! and stops at the *first* candidate that passes. A ladder applies the epoch's
//! one accumulation at every configured step scale, scores the whole grid in a
//! single `rust_scorer` call, and keeps the **best** improvement — so these
//! tests drive `run_train` with a stub scorer that prefers a rung the
//! first-improver search would never have reached.
//!
//! The `rust_scorer` binary is a legitimate boundary to fake (see
//! `scorer_boundary.rs`): the stub scores a whole *directory* of creatures, as
//! the real scorer does, and records both how many times it was invoked and how
//! many creatures it was handed.

#![cfg(unix)]

use neat_ai_backpropagation::backprop::{ApplyOptions, BackpropConfig};
use neat_ai_backpropagation::ladder::{
    DEFAULT_STEP_SCALE_LADDER, DEFAULT_STEP_SCALE_LADDER_CSV, parse_step_scale_ladder,
};
use neat_ai_backpropagation::train::{
    AcceptReason, AcceptanceMode, ScorerAcceptance, TrainCandidateRecord, TrainCreature,
    TrainEpochRecord, TrainJournalHeader, TrainRequest, TrainResult, run_train,
};
use neat_ai_backpropagation::trust_region::TrustRegion;
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// Identity chain `input-0 → h1 → o1`, every gene at 1.0 and every bias at 0.
const IDENTITY_CHAIN: &str = r#"{
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

/// A prepared run: creature, one-record corpus and a batch-capable stub scorer.
struct Fixture {
    _dir: TempDir,
    root: PathBuf,
    creature: PathBuf,
    data: PathBuf,
    scorer: PathBuf,
    calls: PathBuf,
    scored: PathBuf,
}

impl Fixture {
    /// Build a fixture whose single record maps `1.0` to `target`, with a stub
    /// scorer handing back `scores` in creature order (the last value repeats).
    fn new(target: f32, scores: &[&str]) -> Self {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let data = root.join("data");
        fs::create_dir_all(&data).expect("create data dir");
        let mut f = fs::File::create(data.join("0.bin")).expect("create record file");
        f.write_all(&1.0f32.to_le_bytes()).expect("write input");
        f.write_all(&target.to_le_bytes()).expect("write target");

        let creature = root.join("creature.json");
        fs::write(&creature, IDENTITY_CHAIN).expect("write creature");

        let scores_file = root.join("scores.txt");
        fs::write(&scores_file, format!("{}\n", scores.join("\n"))).expect("write scores");
        let calls = root.join("calls.txt");
        let scored = root.join("scored.txt");
        let scorer = write_batch_stub_scorer(&root, &scores_file, &calls, &scored);

        Self {
            _dir: dir,
            root,
            creature,
            data,
            scorer,
            calls,
            scored,
        }
    }

    /// How many times the stub scorer *process* was invoked.
    fn scorer_calls(&self) -> usize {
        count_lines(&self.calls)
    }

    /// How many creatures the stub scorer was handed in total.
    fn creatures_scored(&self) -> usize {
        count_lines(&self.scored)
    }

    /// Journal lines of the given `kind` from a finished run.
    fn journal_lines(&self, out: &Path, kind: &str) -> Vec<String> {
        fs::read_to_string(out.join("journal.jsonl"))
            .expect("journal written")
            .lines()
            .filter(|line| line.contains(&format!("\"kind\":\"{kind}\"")))
            .map(str::to_string)
            .collect()
    }

    /// Parsed candidate lines of a finished run, in journal order.
    fn candidates(&self, out: &Path) -> Vec<TrainCandidateRecord> {
        self.journal_lines(out, "candidate")
            .iter()
            .map(|line| serde_json::from_str(line).expect("candidate line"))
            .collect()
    }

    /// The single epoch line of a one-epoch run.
    fn epoch(&self, out: &Path) -> TrainEpochRecord {
        serde_json::from_str(&self.journal_lines(out, "epoch")[0]).expect("epoch line")
    }
}

/// Count the lines of a tally file a stub scorer appends to.
///
/// A missing file is zero; any other read error is a broken fixture and must
/// not be reported as "the scorer never ran" — several tests assert exactly
/// that.
fn count_lines(path: &Path) -> usize {
    match fs::read_to_string(path) {
        Ok(text) => text.lines().count(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => panic!("cannot read {}: {e}", path.display()),
    }
}

/// Write an executable `/bin/sh` stub standing in for a batch `rust_scorer`.
///
/// Like the real binary it scores every creature in the directory it is handed
/// and prints one `stem → result` entry per file. Each invocation appends to
/// `calls` and each scored creature appends to `scored`, so a test can tell one
/// batch of seven apart from seven single calls.
fn write_batch_stub_scorer(dir: &Path, scores: &Path, calls: &Path, scored: &Path) -> PathBuf {
    let path = dir.join("stub-scorer");
    let body = format!(
        r#"#!/bin/sh
set -eu
printf 'call\n' >> '{calls}'
out=''
for candidate in "$1"/*.json; do
  [ -e "$candidate" ] || continue
  printf 'creature\n' >> '{scored}'
  n=$(wc -l < '{scored}' | tr -d ' ')
  score=$(sed -n "${{n}}p" '{scores}')
  if [ -z "$score" ]; then
    score=$(tail -n 1 '{scores}')
  fi
  stem=$(basename "$candidate" .json)
  if [ -n "$out" ]; then
    out="$out,"
  fi
  out="$out\"$stem\":{{\"score\":$score,\"error\":0.0}}"
done
printf '{{%s}}\n' "$out"
"#,
        calls = calls.display(),
        scores = scores.display(),
        scored = scored.display(),
    );
    fs::write(&path, body).expect("write stub scorer");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod stub scorer");
    path
}

/// Scorer-guided acceptance with the default epsilon and no MSE pre-screen.
fn scorer_guided() -> AcceptanceMode {
    AcceptanceMode::Scorer(ScorerAcceptance::default())
}

/// Run `epochs` epochs over `fixture` with the given ladder and acceptance.
fn train(
    fixture: &Fixture,
    out_name: &str,
    acceptance: AcceptanceMode,
    ladder: &[f64],
    epochs: u64,
) -> Result<TrainResult, String> {
    let out = fixture.root.join(out_name);
    run_train(TrainRequest {
        creature: TrainCreature::Path(&fixture.creature),
        training_data: &fixture.data,
        config: &BackpropConfig::default(),
        epochs,
        max_records: Some(1),
        seed: 1,
        disable_random_samples: false,
        output_dir: &out,
        scorer: Some(&fixture.scorer),
        apply: ApplyOptions::default(),
        acceptance,
        accept_always: false,
        // Deliberately generous: the ladder's rungs are the epoch's attempts,
        // so the halving budget must not add attempts of its own.
        max_backtracks: 6,
        step_scale_ladder: ladder,
        trust_region: TrustRegion::default(),
        trace_store: None,
    })
}

/// Every rung is applied from the one accumulation, the whole grid is scored in
/// a **single** scorer invocation, and every rung is journalled with both its
/// MSE and its score.
#[test]
fn the_whole_ladder_is_scored_in_one_call_and_journalled() {
    let fixture = Fixture::new(2.0, &["0.5", "0.6", "0.9", "0.7"]);
    let ladder = [0.001, 0.005, 0.01];

    let result = train(&fixture, "out", scorer_guided(), &ladder, 1).expect("train");

    // Baseline, then one batch for the whole ladder — not one call per rung.
    assert_eq!(fixture.scorer_calls(), 2);
    assert_eq!(fixture.creatures_scored(), 1 + ladder.len());

    let out = fixture.root.join("out");
    let candidates = fixture.candidates(&out);
    assert_eq!(candidates.len(), ladder.len());
    for (index, rec) in candidates.iter().enumerate() {
        assert_eq!(rec.attempt, index as u32);
        assert_eq!(rec.epoch, 1);
        assert!((rec.step_scale - ladder[index]).abs() < 1e-15);
        assert_eq!(rec.baseline_score, 0.5);
        // Score *and* MSE for every step of the ladder.
        assert!(rec.candidate_score.is_some(), "rung {index} was not scored");
        assert!(rec.candidate_mse.is_finite());
        assert!((rec.mse_delta - (rec.candidate_mse - rec.incumbent_mse)).abs() < 1e-15);
    }
    assert_eq!(
        candidates
            .iter()
            .map(|rec| rec.candidate_score)
            .collect::<Vec<_>>(),
        vec![Some(0.6), Some(0.9), Some(0.7)]
    );
    // The grid itself is journalled, so a remote runner can audit which rungs
    // an epoch scored.
    let header: TrainJournalHeader =
        serde_json::from_str(&fixture.journal_lines(&out, "runHeader")[0]).expect("header line");
    assert_eq!(header.step_scale_ladder, Some(ladder.to_vec()));
    assert_eq!(result.baseline_score.expect("baseline scored").score, 0.5);
}

/// The point of the issue: the ladder keeps the **best** scorer improvement,
/// not the first rung that clears the epsilon.
#[test]
fn the_best_rung_wins_not_the_first_improver() {
    // Every rung improves on the 0.5 baseline, so a first-improver search would
    // stop on the smallest step (0.6). The best is the last rung.
    let fixture = Fixture::new(2.0, &["0.5", "0.6", "0.7", "0.8"]);
    let ladder = [0.001, 0.005, 0.01];

    let result = train(&fixture, "out", scorer_guided(), &ladder, 1).expect("train");

    assert_eq!(result.accepted_epochs, 1);
    assert_eq!(result.best_score.expect("best scored").score, 0.8);
    let out = fixture.root.join("out");
    let epoch = fixture.epoch(&out);
    assert!(epoch.accepted);
    assert_eq!(epoch.accept_reason, AcceptReason::ScoreImproved);
    assert!((epoch.step_scale - 0.01).abs() < 1e-15);
    assert_eq!(epoch.candidate_score, Some(0.8));
    assert_eq!(epoch.ladder_rungs, Some(3));
    // The ladder replaces the halving search, so nothing was backtracked.
    assert_eq!(epoch.backtracks, 0);
    // Only the winner is kept; the rungs that scored well but lost say so.
    let reasons: Vec<AcceptReason> = fixture
        .candidates(&out)
        .iter()
        .map(|rec| rec.accept_reason)
        .collect();
    assert_eq!(
        reasons,
        vec![
            AcceptReason::ScoreNotBest,
            AcceptReason::ScoreNotBest,
            AcceptReason::ScoreImproved,
        ]
    );
}

/// The same fixture through both searches: the halving line search keeps the
/// first candidate the scorer accepts, the ladder keeps the best one.
#[test]
fn the_ladder_beats_the_line_search_on_the_same_scores() {
    let scores = ["0.5", "0.6", "0.7", "0.8"];

    let search_run = Fixture::new(2.0, &scores);
    let with_search = train(&search_run, "out", scorer_guided(), &[], 1).expect("train");
    let search_score = with_search.best_score.expect("best scored").score;
    // The full step already improves on 0.5, so the search never looks further.
    assert_eq!(search_score, 0.6);
    assert_eq!(search_run.epoch(&search_run.root.join("out")).backtracks, 0);

    let ladder_run = Fixture::new(2.0, &scores);
    let with_ladder = train(
        &ladder_run,
        "out",
        scorer_guided(),
        &[0.001, 0.005, 0.01],
        1,
    )
    .expect("train");
    let ladder_score = with_ladder.best_score.expect("best scored").score;

    assert_eq!(ladder_score, 0.8);
    assert!(
        ladder_score > search_score,
        "the ladder must keep the best scorer winner: {ladder_score} vs {search_score}"
    );
}

/// A middle rung can win: neither the smallest nor the largest step is special,
/// only the score is.
#[test]
fn a_middle_rung_can_win() {
    let fixture = Fixture::new(2.0, &["0.5", "0.6", "0.9", "0.7"]);
    let ladder = [0.001, 0.005, 0.01];

    let result = train(&fixture, "out", scorer_guided(), &ladder, 1).expect("train");

    assert_eq!(result.best_score.expect("best scored").score, 0.9);
    let epoch = fixture.epoch(&fixture.root.join("out"));
    assert!((epoch.step_scale - 0.005).abs() < 1e-15);
}

/// No winner leaves the incumbent exactly where it was — and still journals the
/// whole grid, so a dry epoch is auditable.
#[test]
fn no_rung_beating_the_incumbent_leaves_it_unchanged() {
    let fixture = Fixture::new(2.0, &["0.5", "0.4", "0.3", "0.2"]);
    let ladder = [0.001, 0.005, 0.01];

    let result = train(&fixture, "out", scorer_guided(), &ladder, 1).expect("train");

    assert_eq!(result.accepted_epochs, 0);
    assert_eq!(result.best_score.expect("best scored").score, 0.5);
    // MSE fell on every rung and still nothing was kept — the scorer decides.
    assert!((result.best_mse - result.baseline_mse).abs() < 1e-15);
    assert_eq!(
        result.best_json,
        fs::read_to_string(fixture.root.join("out").join("best.json")).expect("best.json")
    );
    let out = fixture.root.join("out");
    let candidates = fixture.candidates(&out);
    assert_eq!(candidates.len(), 3);
    assert!(candidates.iter().all(|rec| !rec.accepted));
    // The closest rung is judged against the epsilon; the rest lost to it.
    assert_eq!(
        candidates
            .iter()
            .map(|rec| rec.accept_reason)
            .collect::<Vec<_>>(),
        vec![
            AcceptReason::ScoreNotImproved,
            AcceptReason::ScoreNotBest,
            AcceptReason::ScoreNotBest,
        ]
    );
    assert!(!fixture.epoch(&out).accepted);
}

/// Equal scores keep the smaller step: the ladder is ascending and a tie is no
/// reason to move further than the evidence supports.
#[test]
fn a_tie_keeps_the_smaller_step() {
    let fixture = Fixture::new(2.0, &["0.5", "0.9", "0.9"]);
    let ladder = [0.001, 0.01];

    train(&fixture, "out", scorer_guided(), &ladder, 1).expect("train");

    let epoch = fixture.epoch(&fixture.root.join("out"));
    assert!((epoch.step_scale - 0.001).abs() < 1e-15);
}

/// The optional pre-screen is the issue's catastrophic rejection: a rung whose
/// slice MSE did not fall never reaches the batch.
#[test]
fn the_mse_pre_screen_drops_rungs_before_the_batch() {
    // target == the creature's own output, so MSE is already 0 and no rung can
    // lower it.
    let fixture = Fixture::new(1.0, &["0.5", "9.0", "9.0"]);
    let ladder = [0.001, 0.01];

    let result = train(
        &fixture,
        "out",
        AcceptanceMode::Scorer(ScorerAcceptance {
            mse_pre_screen: true,
            ..ScorerAcceptance::default()
        }),
        &ladder,
        1,
    )
    .expect("train");

    assert_eq!(result.accepted_epochs, 0);
    // Baseline only — the batch was never spawned, even though the stub's next
    // payloads would have been easy wins.
    assert_eq!(fixture.scorer_calls(), 1);
    assert_eq!(fixture.creatures_scored(), 1);
    let candidates = fixture.candidates(&fixture.root.join("out"));
    assert_eq!(candidates.len(), 2);
    for rec in &candidates {
        assert_eq!(rec.accept_reason, AcceptReason::MsePreScreenRejected);
        assert_eq!(rec.candidate_score, None);
        assert_eq!(rec.score_delta, None);
        // MSE is still journalled for every step, screened out or not.
        assert!(rec.candidate_mse.is_finite());
    }
}

/// A one-rung ladder and the line search's first attempt are the same
/// candidate: both come from the epoch's single accumulation applied at the
/// same step scale.
#[test]
fn a_single_rung_ladder_matches_the_line_search_candidate() {
    let ladder_run = Fixture::new(2.0, &["0.5", "0.9"]);
    let ladder = [ApplyOptions::default().step_scale];
    let with_ladder = train(&ladder_run, "out", scorer_guided(), &ladder, 1).expect("train");

    let search_run = Fixture::new(2.0, &["0.5", "0.9"]);
    let with_search = train(&search_run, "out", scorer_guided(), &[], 1).expect("train");

    assert_eq!(with_ladder.accepted_epochs, with_search.accepted_epochs);
    assert!((with_ladder.best_mse - with_search.best_mse).abs() < 1e-15);
    assert_eq!(with_ladder.best_json, with_search.best_json);
    let ladder_epoch = ladder_run.epoch(&ladder_run.root.join("out"));
    let search_epoch = search_run.epoch(&search_run.root.join("out"));
    assert!((ladder_epoch.after_mse - search_epoch.after_mse).abs() < 1e-15);
    assert_eq!(ladder_epoch.ladder_rungs, Some(1));
    assert_eq!(search_epoch.ladder_rungs, None);
}

/// Across epochs the winner becomes the incumbent every later rung is judged
/// against — one accumulation, one batch, one accept per epoch.
#[test]
fn each_epoch_judges_the_ladder_against_the_last_winner() {
    let fixture = Fixture::new(2.0, &["0.5", "0.6", "0.7", "0.8", "0.9"]);
    let ladder = [0.001, 0.01];

    let result = train(&fixture, "out", scorer_guided(), &ladder, 2).expect("train");

    assert_eq!(result.accepted_epochs, 2);
    assert_eq!(result.best_score.expect("best scored").score, 0.9);
    // Baseline plus one batch per epoch.
    assert_eq!(fixture.scorer_calls(), 3);
    let out = fixture.root.join("out");
    let candidates = fixture.candidates(&out);
    assert_eq!(candidates.len(), 4);
    // Epoch 1 judged against the run baseline, epoch 2 against epoch 1's winner.
    assert_eq!(candidates[0].baseline_score, 0.5);
    assert_eq!(candidates[3].baseline_score, 0.7);
}

/// A ladder on an MSE run is a misconfiguration: the grid would be ignored, so
/// the run would look configured while judging on MSE.
#[test]
fn a_ladder_without_scorer_guided_acceptance_is_refused() {
    let fixture = Fixture::new(2.0, &["0.5"]);

    let err =
        train(&fixture, "out", AcceptanceMode::Mse, &[0.001, 0.01], 1).expect_err("must refuse");

    assert!(err.contains("step-scale ladder"), "unexpected error: {err}");
    assert_eq!(fixture.scorer_calls(), 0, "refuse before scoring");
}

/// A rung the applier would silently rewrite is refused before any work — the
/// journal must never record a step scale that was not the one applied.
#[test]
fn an_unusable_rung_is_refused_before_any_scoring() {
    for bad in [0.0, -0.001, f64::NAN, 1.5] {
        let fixture = Fixture::new(2.0, &["0.5"]);
        let err =
            train(&fixture, "out", scorer_guided(), &[0.001, bad], 1).expect_err("must refuse");
        assert!(
            err.contains("step-scale ladder rungs"),
            "{bad}: unexpected error: {err}"
        );
        assert_eq!(fixture.scorer_calls(), 0, "{bad}: refuse before scoring");
    }
}

/// The default grid is the issue's own ladder, and the CLI spelling parses back
/// to it.
#[test]
fn the_default_grid_is_configurable_from_the_cli_spelling() {
    assert_eq!(
        parse_step_scale_ladder(DEFAULT_STEP_SCALE_LADDER_CSV).expect("default grid parses"),
        DEFAULT_STEP_SCALE_LADDER.to_vec()
    );
    let custom = parse_step_scale_ladder("0.002, 0.02").expect("custom grid parses");
    assert_eq!(custom, vec![0.002, 0.02]);
}

/// Acceptance criterion: the grid is configurable from `train --help`.
#[test]
fn train_help_documents_the_step_scale_ladder_flag() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_neat_ai_backpropagation"))
        .args(["train", "--help"])
        .output()
        .expect("run train --help");
    assert!(output.status.success(), "train --help exits cleanly");
    let help = String::from_utf8(output.stdout).expect("utf-8 help text");
    assert!(
        help.contains("--step-scale-ladder"),
        "train --help documents --step-scale-ladder:\n{help}"
    );
}
