//! Observable behaviour of [`score_creature`] at the process boundary (issue #24).
//!
//! `score_creature` is the authoritative accept gate in the production win
//! protocol — `run_train` calls it whenever `--scorer` is set — yet nothing
//! exercised it end to end, so the candidate-directory layout, the non-zero
//! exit path and the stdout fallbacks could all break silently.
//!
//! The `rust_scorer` binary is a legitimate boundary to fake: each test writes
//! a tiny stub executable that prints a fixed payload, then asserts on the
//! `ScoreResult` (or the error string) that `score_creature` returns. Nothing
//! here asserts *how* the process is invoked beyond the directory contract the
//! real scorer relies on.

#![cfg(unix)]

use neat_ai_backpropagation::{ScoreResult, score_creature};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

const CREATURE_JSON: &str = r#"{"semanticVersion":"4.0.0","input":1,"output":1}"#;

/// Write an executable `/bin/sh` stub standing in for `rust_scorer`.
fn write_stub(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write stub scorer");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod stub scorer");
    path
}

/// A temp workspace plus a training-data file for the scorer's second argument.
fn workspace() -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let training = dir.path().join("training.bin");
    fs::write(&training, b"training-bytes").expect("write training data");
    (dir, training)
}

#[test]
fn returns_the_scored_creature_from_map_stdout() {
    let (dir, training) = workspace();
    let scorer = write_stub(
        dir.path(),
        "map-scorer",
        r#"printf '%s\n' '{"trained":{"score":0.5,"error":0.25,"complexityPenalty":0.01}}'"#,
    );

    let result = score_creature(&scorer, CREATURE_JSON, &training, dir.path())
        .expect("stub scorer should score the candidate");

    assert_eq!(
        result,
        ScoreResult {
            score: 0.5,
            error: 0.25,
            complexity_penalty: 0.01,
        }
    );
}

#[test]
fn falls_back_to_a_single_result_object_without_a_stem_key() {
    let (dir, training) = workspace();
    let scorer = write_stub(
        dir.path(),
        "object-scorer",
        r#"printf '%s\n' '{"score":-1.5,"error":2.25}'"#,
    );

    let result = score_creature(&scorer, CREATURE_JSON, &training, dir.path())
        .expect("single-object stdout should still yield a score");

    assert_eq!(result.score, -1.5);
    assert_eq!(result.error, 2.25);
    // `complexityPenalty` is optional and defaults to zero.
    assert_eq!(result.complexity_penalty, 0.0);
}

#[test]
fn ignores_scorer_log_lines_printed_before_the_result() {
    let (dir, training) = workspace();
    let scorer = write_stub(
        dir.path(),
        "chatty-scorer",
        "printf '%s\\n' 'loading corpus…' '' '{\"trained\":{\"score\":0.75,\"error\":0.125}}'",
    );

    let result = score_creature(&scorer, CREATURE_JSON, &training, dir.path())
        .expect("trailing JSON line should be parsed past the log noise");

    assert_eq!(result.score, 0.75);
}

#[test]
fn hands_the_creature_to_the_scorer_as_trained_json_in_a_candidate_directory() {
    let (dir, training) = workspace();
    // The stub echoes back what it was handed so the layout contract is
    // observable: argv[1] is a directory holding `trained.json`, argv[2] is the
    // training data.
    let scorer = write_stub(
        dir.path(),
        "echo-scorer",
        concat!(
            "cat \"$1/trained.json\" > \"$1/../seen-creature.json\"\n",
            "printf '%s\\n' \"$2\" > \"$1/../seen-training.txt\"\n",
            "printf '%s\\n' '{\"trained\":{\"score\":1.0,\"error\":0.0}}'"
        ),
    );

    let result = score_creature(&scorer, CREATURE_JSON, &training, dir.path())
        .expect("stub scorer should read the candidate it was handed");

    assert_eq!(result.score, 1.0);
    assert_eq!(
        fs::read_to_string(dir.path().join("scorer-candidate").join("trained.json"))
            .expect("candidate creature should be left on disk"),
        CREATURE_JSON
    );
    assert_eq!(
        fs::read_to_string(dir.path().join("seen-creature.json"))
            .expect("scorer read the creature"),
        CREATURE_JSON
    );
    assert_eq!(
        fs::read_to_string(dir.path().join("seen-training.txt"))
            .expect("scorer received the training data")
            .trim(),
        training.to_string_lossy()
    );
}

#[test]
fn reports_a_non_zero_exit_with_its_status_and_stdout() {
    let (dir, training) = workspace();
    let scorer = write_stub(
        dir.path(),
        "failing-scorer",
        "printf '%s\\n' 'corpus unreadable'\nexit 3",
    );

    let err = score_creature(&scorer, CREATURE_JSON, &training, dir.path())
        .expect_err("a non-zero exit must not be reported as a score");

    assert!(err.contains("scorer exited"), "unexpected error: {err}");
    assert!(
        err.contains('3'),
        "error should name the exit status: {err}"
    );
    assert!(
        err.contains("corpus unreadable"),
        "error should carry the scorer's stdout: {err}"
    );
}

#[test]
fn reports_an_empty_score_map_as_no_creature_scores() {
    let (dir, training) = workspace();
    let scorer = write_stub(dir.path(), "empty-scorer", r#"printf '%s\n' '{}'"#);

    let err = score_creature(&scorer, CREATURE_JSON, &training, dir.path())
        .expect_err("an empty result map is not a score");

    assert_eq!(err, "scorer returned no creature scores");
}

#[test]
fn reports_unparsable_stdout() {
    let (dir, training) = workspace();
    let scorer = write_stub(dir.path(), "babbling-scorer", "printf '%s\\n' 'not json'");

    let err = score_creature(&scorer, CREATURE_JSON, &training, dir.path())
        .expect_err("stdout without JSON is not a score");

    assert!(
        err.starts_with("could not parse scorer stdout"),
        "unexpected error: {err}"
    );
    assert!(err.contains("not json"), "error should quote stdout: {err}");
}

#[test]
fn reports_a_scorer_binary_that_cannot_be_spawned() {
    let (dir, training) = workspace();
    let missing = dir.path().join("no-such-scorer");

    let err = score_creature(&missing, CREATURE_JSON, &training, dir.path())
        .expect_err("a missing scorer binary must fail loudly");

    assert!(
        err.starts_with("failed to spawn scorer"),
        "unexpected error: {err}"
    );
}
