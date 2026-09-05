//! Scorer-guided acceptance: the scorer is the judge, not slice MSE (issue #104).
//!
//! `run_train` historically accepted an epoch on `after_mse < best_mse` and only
//! invoked `rust_scorer` after the loop. On a highly evolved production creature
//! a candidate can lower training-slice MSE and still lower the fitness score
//! evolution is actually optimising, so these tests drive `run_train` with a
//! stub scorer whose verdict *disagrees* with MSE and assert the trainer follows
//! the scorer.
//!
//! The `rust_scorer` binary is a legitimate boundary to fake (see
//! `scorer_boundary.rs`): each stub prints the next score from a list and
//! records that it was called, so both the verdict and the number of scorer
//! invocations are observable.

#![cfg(unix)]

use neat_ai_backpropagation::backprop::{ApplyOptions, BackpropConfig};
use neat_ai_backpropagation::train::{
    AcceptReason, AcceptanceMode, DEFAULT_MIN_SCORE_IMPROVEMENT, ScorerAcceptance,
    TrainCandidateRecord, TrainCreature, TrainEpochRecord, TrainJournalHeader, TrainRequest,
    TrainResult, resolve_acceptance, run_train,
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

/// A prepared run: creature, one-record corpus and a stub scorer.
struct Fixture {
    _dir: TempDir,
    root: PathBuf,
    creature: PathBuf,
    data: PathBuf,
    scorer: PathBuf,
    calls: PathBuf,
}

impl Fixture {
    /// Build a fixture whose single record maps `input` to `target`, with a
    /// stub scorer handing back `scores` in call order (the last value repeats).
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
        let scorer = write_stub_scorer(&root, &scores_file, &calls);

        Self {
            _dir: dir,
            root,
            creature,
            data,
            scorer,
            calls,
        }
    }

    /// How many times the stub scorer was invoked so far.
    ///
    /// A missing file is zero calls; any other read error is a broken fixture
    /// and must not be reported as "the scorer never ran" — several tests
    /// assert exactly that.
    fn scorer_calls(&self) -> usize {
        match fs::read_to_string(&self.calls) {
            Ok(text) => text.lines().count(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
            Err(e) => panic!("cannot read {}: {e}", self.calls.display()),
        }
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
}

/// Write an executable `/bin/sh` stub standing in for `rust_scorer`.
///
/// Each call appends to `calls` and prints the matching line of `scores`, so a
/// test can make the scorer disagree with MSE and still count invocations.
fn write_stub_scorer(dir: &Path, scores: &Path, calls: &Path) -> PathBuf {
    let path = dir.join("stub-scorer");
    let body = format!(
        r#"#!/bin/sh
set -eu
printf 'call\n' >> '{calls}'
n=$(wc -l < '{calls}' | tr -d ' ')
score=$(sed -n "${{n}}p" '{scores}')
if [ -z "$score" ]; then
  score=$(tail -n 1 '{scores}')
fi
printf '{{"trained":{{"score":%s,"error":0.0}}}}\n' "$score"
"#,
        calls = calls.display(),
        scores = scores.display(),
    );
    fs::write(&path, body).expect("write stub scorer");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod stub scorer");
    path
}

/// Run one epoch over `fixture` with the given acceptance mode, single attempt.
fn train(
    fixture: &Fixture,
    out_name: &str,
    acceptance: AcceptanceMode,
    with_scorer: bool,
) -> Result<TrainResult, String> {
    train_with_backtracks(fixture, out_name, acceptance, with_scorer, 0)
}

/// Run one epoch, allowing the backtracking line search `max_backtracks` retries.
fn train_with_backtracks(
    fixture: &Fixture,
    out_name: &str,
    acceptance: AcceptanceMode,
    with_scorer: bool,
    max_backtracks: u32,
) -> Result<TrainResult, String> {
    let out = fixture.root.join(out_name);
    run_train(TrainRequest {
        creature: TrainCreature::Path(&fixture.creature),
        training_data: &fixture.data,
        config: &BackpropConfig::default(),
        epochs: 1,
        max_records: Some(1),
        seed: 1,
        disable_random_samples: false,
        output_dir: &out,
        scorer: with_scorer.then_some(fixture.scorer.as_path()),
        apply: ApplyOptions::default(),
        acceptance,
        accept_always: false,
        max_backtracks,
        step_scale_ladder: &[],
        trust_region: TrustRegion::default(),
        trace_store: None,
    })
}

/// Scorer-guided acceptance with the default epsilon and no MSE pre-screen.
fn scorer_guided() -> AcceptanceMode {
    AcceptanceMode::Scorer(ScorerAcceptance::default())
}

/// The MSE-only path is unchanged: a candidate that lowers slice MSE is kept
/// even when the scorer (present only for the before/after report) says the
/// score fell.
#[test]
fn mse_only_mode_still_accepts_on_slice_mse_alone() {
    let fixture = Fixture::new(2.0, &["0.5", "0.1"]);

    let result = train(&fixture, "out", AcceptanceMode::Mse, true).expect("train");

    assert_eq!(
        result.accepted_epochs, 1,
        "the fixture's candidate must lower slice MSE for the rest of this suite to mean anything"
    );
    assert!(result.best_mse < result.baseline_mse);
    // Scoring stays a post-loop report in MSE mode — baseline, then best.
    assert_eq!(fixture.scorer_calls(), 2);
    let epochs = fixture.journal_lines(&fixture.root.join("out"), "epoch");
    let epoch: TrainEpochRecord = serde_json::from_str(&epochs[0]).expect("epoch line");
    assert_eq!(epoch.accept_reason, AcceptReason::MseImproved);
    assert!(epoch.candidate_score.is_none());
    // MSE mode journals no per-candidate lines.
    assert!(
        fixture
            .journal_lines(&fixture.root.join("out"), "candidate")
            .is_empty()
    );
}

/// The point of the issue: the same candidate the MSE loop keeps is rolled back
/// when the scorer says the fitness fell.
#[test]
fn scorer_guided_rolls_back_a_candidate_whose_score_falls() {
    let fixture = Fixture::new(2.0, &["0.5", "0.1"]);

    let result = train(&fixture, "out", scorer_guided(), true).expect("train");

    assert_eq!(result.accepted_epochs, 0, "a lower score is not a win");
    // MSE is still measured and reported — it just does not decide.
    assert!((result.best_mse - result.baseline_mse).abs() < 1e-15);
    let out = fixture.root.join("out");
    let candidates = fixture.journal_lines(&out, "candidate");
    assert_eq!(candidates.len(), 1);
    let rec: TrainCandidateRecord = serde_json::from_str(&candidates[0]).expect("candidate line");
    assert_eq!(rec.accept_reason, AcceptReason::ScoreNotImproved);
    assert!(!rec.accepted);
    assert_eq!(rec.baseline_score, 0.5);
    assert_eq!(rec.candidate_score, Some(0.1));
    assert!((rec.score_delta.expect("score delta") + 0.4).abs() < 1e-12);
    // The MSE delta is journalled as a diagnostic beside the scorer verdict.
    assert!(
        rec.mse_delta < 0.0,
        "candidate lowered MSE yet lost on score: {rec:?}"
    );
    assert!(rec.step_scale > 0.0);
}

/// A candidate the scorer likes is kept, and the run reports the scorer's own
/// numbers rather than re-scoring the winner.
#[test]
fn scorer_guided_accepts_a_candidate_whose_score_rises() {
    let fixture = Fixture::new(2.0, &["0.5", "0.6"]);

    let result = train(&fixture, "out", scorer_guided(), true).expect("train");

    assert_eq!(result.accepted_epochs, 1);
    assert_eq!(result.baseline_score.expect("baseline scored").score, 0.5);
    assert_eq!(result.best_score.expect("best scored").score, 0.6);
    // Baseline once, candidate once — the accepted candidate is not re-scored.
    assert_eq!(fixture.scorer_calls(), 2);
    let out = fixture.root.join("out");
    let epoch: TrainEpochRecord =
        serde_json::from_str(&fixture.journal_lines(&out, "epoch")[0]).expect("epoch line");
    assert!(epoch.accepted);
    assert_eq!(epoch.accept_reason, AcceptReason::ScoreImproved);
    assert_eq!(epoch.baseline_score, Some(0.5));
    assert_eq!(epoch.candidate_score, Some(0.6));
    assert!((epoch.score_delta.expect("score delta") - 0.1).abs() < 1e-12);
}

/// The baseline is scored before any candidate is judged, so epoch 1 has a real
/// incumbent score to beat rather than a post-hoc comparison.
#[test]
fn the_baseline_is_scored_before_the_first_candidate() {
    let fixture = Fixture::new(2.0, &["0.5", "0.6"]);

    let result = train(&fixture, "out", scorer_guided(), true).expect("train");

    let out = fixture.root.join("out");
    let rec: TrainCandidateRecord =
        serde_json::from_str(&fixture.journal_lines(&out, "candidate")[0]).expect("candidate line");
    // The first stub payload (0.5) is the incumbent the candidate was judged
    // against — it cannot be, unless the baseline ran first.
    assert_eq!(rec.baseline_score, 0.5);
    assert_eq!(result.baseline_score.expect("baseline scored").score, 0.5);
    assert!(
        out.join("scorer-work").join("baseline").is_dir(),
        "baseline scorer work directory should exist"
    );
    // The run's own opening number is journalled under its own name — a
    // candidate line's `baselineScore` is the incumbent, which moves on accept.
    let header: TrainJournalHeader =
        serde_json::from_str(&fixture.journal_lines(&out, "runHeader")[0]).expect("header line");
    assert_eq!(header.baseline_score, Some(0.5));
    assert_eq!(
        header.acceptance,
        AcceptanceMode::Scorer(ScorerAcceptance::default())
    );
}

/// An MSE run handed scorer knobs is a misconfiguration: the flags would be
/// ignored, so the run would look configured while judging on MSE.
#[test]
fn scorer_settings_on_an_mse_run_are_refused() {
    assert_eq!(
        resolve_acceptance(false, DEFAULT_MIN_SCORE_IMPROVEMENT, false).unwrap(),
        AcceptanceMode::Mse
    );
    assert!(
        resolve_acceptance(false, DEFAULT_MIN_SCORE_IMPROVEMENT, true)
            .unwrap_err()
            .contains("msePreScreen")
    );
    assert!(
        resolve_acceptance(false, 0.5, false)
            .unwrap_err()
            .contains("minScoreImprovement")
    );
    assert_eq!(
        resolve_acceptance(true, 0.5, true).unwrap(),
        AcceptanceMode::Scorer(ScorerAcceptance {
            min_improvement: 0.5,
            mse_pre_screen: true,
        })
    );
}

/// The epsilon is a real gate: the conservative default rejects a gain in the
/// noise, and lowering it accepts the same candidate.
#[test]
fn a_gain_below_the_minimum_improvement_is_not_a_win() {
    let default_run = Fixture::new(2.0, &["0.5", "0.500000001"]);
    let rejected = train(&default_run, "out", scorer_guided(), true).expect("train");
    assert_eq!(rejected.accepted_epochs, 0);
    let rec: TrainCandidateRecord = serde_json::from_str(
        &default_run.journal_lines(&default_run.root.join("out"), "candidate")[0],
    )
    .expect("candidate line");
    assert_eq!(rec.accept_reason, AcceptReason::ScoreNotImproved);
    // The default has to be conservative for the rejection above to mean
    // "below the epsilon" rather than "below zero".
    const { assert!(DEFAULT_MIN_SCORE_IMPROVEMENT > 1e-9) };

    let lenient_run = Fixture::new(2.0, &["0.5", "0.500000001"]);
    let accepted = train(
        &lenient_run,
        "out",
        AcceptanceMode::Scorer(ScorerAcceptance {
            min_improvement: 1e-12,
            ..ScorerAcceptance::default()
        }),
        true,
    )
    .expect("train");
    assert_eq!(accepted.accepted_epochs, 1);
}

/// The optional MSE pre-screen keeps the expensive scorer off candidates that
/// did not even lower the cheap diagnostic.
#[test]
fn the_mse_pre_screen_skips_the_scorer_when_slice_mse_did_not_fall() {
    // target == the creature's own output, so MSE is already 0 and no apply can
    // lower it.
    let fixture = Fixture::new(1.0, &["0.5", "9.0"]);

    let result = train(
        &fixture,
        "out",
        AcceptanceMode::Scorer(ScorerAcceptance {
            mse_pre_screen: true,
            ..ScorerAcceptance::default()
        }),
        true,
    )
    .expect("train");

    assert_eq!(result.accepted_epochs, 0);
    // Baseline only — the candidate never reached the scorer, even though the
    // stub's next payload (9.0) would have been an easy win.
    assert_eq!(fixture.scorer_calls(), 1);
    let rec: TrainCandidateRecord =
        serde_json::from_str(&fixture.journal_lines(&fixture.root.join("out"), "candidate")[0])
            .expect("candidate line");
    assert_eq!(rec.accept_reason, AcceptReason::MsePreScreenRejected);
    assert_eq!(rec.candidate_score, None);
    assert_eq!(rec.score_delta, None);
}

/// Every step of the backtracking line search is scored and journalled, not
/// just the candidate the epoch finished on — the halved step is a different
/// creature and the scorer is the only thing that can judge it.
#[test]
fn each_backtracking_step_is_scored_and_journalled() {
    // Baseline 0.5, then three losing candidates: the full step and two
    // halvings all fail the gate, so the epoch exhausts its budget.
    let fixture = Fixture::new(2.0, &["0.5", "0.1", "0.2", "0.3"]);

    let result = train_with_backtracks(&fixture, "out", scorer_guided(), true, 2).expect("train");

    assert_eq!(result.accepted_epochs, 0);
    // Baseline + one scorer run per attempt.
    assert_eq!(fixture.scorer_calls(), 4);
    let candidates: Vec<TrainCandidateRecord> = fixture
        .journal_lines(&fixture.root.join("out"), "candidate")
        .iter()
        .map(|line| serde_json::from_str(line).expect("candidate line"))
        .collect();
    assert_eq!(candidates.len(), 3);
    for (index, rec) in candidates.iter().enumerate() {
        assert_eq!(rec.attempt, index as u32);
        assert_eq!(rec.epoch, 1);
        assert_eq!(rec.accept_reason, AcceptReason::ScoreNotImproved);
        // Each attempt is judged against the same unmoved incumbent.
        assert_eq!(rec.baseline_score, 0.5);
        assert!(rec.candidate_score.is_some());
    }
    assert_eq!(
        candidates
            .iter()
            .map(|r| r.candidate_score)
            .collect::<Vec<_>>(),
        vec![Some(0.1), Some(0.2), Some(0.3)]
    );
    // The step halves on every retry.
    for pair in candidates.windows(2) {
        assert!(
            (pair[1].step_scale - pair[0].step_scale / 2.0).abs() < 1e-15,
            "step should halve: {} -> {}",
            pair[0].step_scale,
            pair[1].step_scale
        );
    }
}

/// A backtracked step the scorer likes is kept, and its score becomes the
/// incumbent the run reports.
#[test]
fn a_backtracked_step_can_win_on_score() {
    let fixture = Fixture::new(2.0, &["0.5", "0.1", "0.9"]);

    let result = train_with_backtracks(&fixture, "out", scorer_guided(), true, 3).expect("train");

    assert_eq!(result.accepted_epochs, 1);
    assert_eq!(result.best_score.expect("best scored").score, 0.9);
    // Baseline, the losing full step, then the winning halved step — the
    // search stops as soon as the scorer accepts.
    assert_eq!(fixture.scorer_calls(), 3);
    let epoch: TrainEpochRecord =
        serde_json::from_str(&fixture.journal_lines(&fixture.root.join("out"), "epoch")[0])
            .expect("epoch line");
    assert_eq!(epoch.backtracks, 1);
    assert_eq!(epoch.accept_reason, AcceptReason::ScoreImproved);
    assert_eq!(epoch.candidate_score, Some(0.9));
}

/// With the pre-screen on, a candidate whose MSE *did* fall still reaches the
/// scorer — the screen filters, it does not decide.
#[test]
fn the_mse_pre_screen_still_lets_an_improving_candidate_reach_the_scorer() {
    let fixture = Fixture::new(2.0, &["0.5", "0.6"]);

    let result = train(
        &fixture,
        "out",
        AcceptanceMode::Scorer(ScorerAcceptance {
            mse_pre_screen: true,
            ..ScorerAcceptance::default()
        }),
        true,
    )
    .expect("train");

    assert_eq!(result.accepted_epochs, 1);
    assert_eq!(fixture.scorer_calls(), 2);
    let rec: TrainCandidateRecord =
        serde_json::from_str(&fixture.journal_lines(&fixture.root.join("out"), "candidate")[0])
            .expect("candidate line");
    assert_eq!(rec.accept_reason, AcceptReason::ScoreImproved);
    assert_eq!(rec.candidate_score, Some(0.6));
}

/// Without the pre-screen the scorer judges a candidate MSE would have dropped.
#[test]
fn without_the_pre_screen_a_worse_mse_candidate_can_still_win_on_score() {
    let fixture = Fixture::new(1.0, &["0.5", "9.0"]);

    let result = train(&fixture, "out", scorer_guided(), true).expect("train");

    assert_eq!(
        result.accepted_epochs, 1,
        "the scorer is the judge — MSE must not veto"
    );
    assert_eq!(result.best_score.expect("best scored").score, 9.0);
    assert_eq!(fixture.scorer_calls(), 2);
}

/// Scorer-guided acceptance without a scorer binary is a configuration error,
/// not a silent fall back to MSE.
#[test]
fn scorer_guided_without_a_scorer_binary_fails_loudly() {
    let fixture = Fixture::new(2.0, &["0.5"]);

    let err = train(&fixture, "out", scorer_guided(), false).expect_err("must refuse");

    assert!(
        err.contains("scorer-guided acceptance"),
        "unexpected error: {err}"
    );
    assert_eq!(fixture.scorer_calls(), 0);
}

/// `accept_always` keeps a candidate regardless of the verdict, which would
/// silently disable the gate — refuse the combination instead.
#[test]
fn accept_always_is_refused_under_scorer_guided_acceptance() {
    let fixture = Fixture::new(2.0, &["0.5", "0.1"]);
    let out = fixture.root.join("out");

    let err = run_train(TrainRequest {
        creature: TrainCreature::Path(&fixture.creature),
        training_data: &fixture.data,
        config: &BackpropConfig::default(),
        epochs: 1,
        max_records: Some(1),
        seed: 1,
        disable_random_samples: false,
        output_dir: &out,
        scorer: Some(&fixture.scorer),
        apply: ApplyOptions::default(),
        acceptance: scorer_guided(),
        accept_always: true,
        max_backtracks: 0,
        step_scale_ladder: &[],
        trust_region: TrustRegion::default(),
        trace_store: None,
    })
    .expect_err("must refuse");

    assert!(err.contains("acceptAlways"), "unexpected error: {err}");
    assert_eq!(fixture.scorer_calls(), 0);
}

/// A negative or non-finite epsilon is a configuration error `run_train`
/// refuses — a `NaN` epsilon would reject every candidate and read as "the
/// scorer found nothing".
#[test]
fn a_negative_or_non_finite_minimum_improvement_is_refused() {
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let fixture = Fixture::new(2.0, &["0.5"]);
        let err = train(
            &fixture,
            "out",
            AcceptanceMode::Scorer(ScorerAcceptance {
                min_improvement: bad,
                ..ScorerAcceptance::default()
            }),
            true,
        )
        .expect_err("must refuse");

        assert!(
            err.contains("minimum score improvement"),
            "{bad}: unexpected error: {err}"
        );
        assert_eq!(fixture.scorer_calls(), 0, "{bad}: refuse before scoring");
    }
}

/// A scorer that dies mid-loop fails the run — a missing verdict is never
/// reconciled as "no improvement".
#[test]
fn a_failing_scorer_fails_the_run() {
    let fixture = Fixture::new(2.0, &["0.5"]);
    // The stub has one score, so the second call prints the repeat — replace it
    // with a scorer that exits non-zero on its second call instead.
    let failing = fixture.root.join("failing-scorer");
    fs::write(
        &failing,
        format!(
            "#!/bin/sh\nset -eu\nprintf 'call\\n' >> '{calls}'\n\
             n=$(wc -l < '{calls}' | tr -d ' ')\n\
             if [ \"$n\" -gt 1 ]; then printf 'corpus unreadable\\n'; exit 4; fi\n\
             printf '{{\"trained\":{{\"score\":0.5,\"error\":0.0}}}}\\n'\n",
            calls = fixture.calls.display()
        ),
    )
    .expect("write failing scorer");
    fs::set_permissions(&failing, fs::Permissions::from_mode(0o755)).expect("chmod");

    let out = fixture.root.join("out");
    let err = run_train(TrainRequest {
        creature: TrainCreature::Path(&fixture.creature),
        training_data: &fixture.data,
        config: &BackpropConfig::default(),
        epochs: 1,
        max_records: Some(1),
        seed: 1,
        disable_random_samples: false,
        output_dir: &out,
        scorer: Some(&failing),
        apply: ApplyOptions::default(),
        acceptance: scorer_guided(),
        accept_always: false,
        max_backtracks: 0,
        step_scale_ladder: &[],
        trust_region: TrustRegion::default(),
        trace_store: None,
    })
    .expect_err("a dead scorer must not be reconciled as a rejection");

    assert!(err.contains("scorer exited"), "unexpected error: {err}");
}
