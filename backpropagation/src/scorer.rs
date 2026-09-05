//! Authoritative `rust_scorer` integration (optional train-time gate).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

/// Parsed fields from a scorer result object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ScoreResult {
    /// Authoritative fitness score (larger-is-better).
    pub score: f64,
    /// Average error (smaller-is-better).
    pub error: f64,
    /// Optional complexity penalty.
    #[serde(default)]
    pub complexity_penalty: f64,
}

/// Stem [`score_creature`] writes its single candidate under.
const SINGLE_CANDIDATE_STEM: &str = "trained";

/// Score a single creature JSON by writing it into a temp directory and
/// invoking `rust_scorer <dir> <training-data>`.
pub fn score_creature(
    scorer: &Path,
    creature_json: &str,
    training_data: &Path,
    work_dir: &Path,
) -> Result<ScoreResult, String> {
    let mut scored = score_creatures(
        scorer,
        &[(SINGLE_CANDIDATE_STEM, creature_json)],
        training_data,
        work_dir,
    )?;
    // One candidate in, one score out — `score_creatures` already refused a
    // response that dropped it.
    Ok(scored.remove(0))
}

/// Score several creatures in **one** `rust_scorer` invocation (issue #106).
///
/// `rust_scorer` takes a *directory* of creatures and prints one result per
/// file stem, so a whole step-scale ladder costs one process launch and one
/// corpus pass instead of one per candidate. Results come back in the order
/// the candidates were supplied.
///
/// `candidates` is `(stem, creature JSON)`; stems must be unique and are the
/// keys the scorer echoes back. A stem the scorer did not report is an error,
/// never a silently dropped candidate.
pub fn score_creatures(
    scorer: &Path,
    candidates: &[(&str, &str)],
    training_data: &Path,
    work_dir: &Path,
) -> Result<Vec<ScoreResult>, String> {
    if candidates.is_empty() {
        return Err("no creatures to score".into());
    }
    let dir = work_dir.join("scorer-candidate");
    // Clear first: the directory is reused across epochs, and a candidate left
    // behind by an earlier, longer ladder would otherwise be scored again and
    // charged to this batch.
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for (stem, creature_json) in candidates {
        fs::write(dir.join(format!("{stem}.json")), creature_json)
            .map_err(|e| format!("{stem}.json: {e}"))?;
    }
    let output = Command::new(scorer)
        .arg(&dir)
        .arg(training_data)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("failed to spawn scorer: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "scorer exited {}: stdout={}",
            output.status,
            String::from_utf8_lossy(&output.stdout)
        ));
    }
    let mut scores = parse_scorer_stdout(&output.stdout)?;
    if scores.is_empty() {
        return Err("scorer returned no creature scores".into());
    }
    // A single candidate is matched by position, not by name: `score_creature`
    // has always accepted whatever stem the scorer chose to echo.
    if candidates.len() == 1 && scores.len() == 1 {
        return Ok(scores.into_values().collect());
    }
    candidates
        .iter()
        .map(|(stem, _)| {
            scores
                .remove(*stem)
                .ok_or_else(|| format!("scorer returned no score for candidate '{stem}'"))
        })
        .collect()
}

fn parse_scorer_stdout(stdout: &[u8]) -> Result<BTreeMap<String, ScoreResult>, String> {
    let text = String::from_utf8_lossy(stdout);
    // rust_scorer prints one JSON object (map of stem → result) or a JSON array.
    for line in text.lines().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(map) = serde_json::from_str::<BTreeMap<String, ScoreResult>>(trimmed) {
            return Ok(map);
        }
        if let Ok(one) = serde_json::from_str::<ScoreResult>(trimmed) {
            let mut map = BTreeMap::new();
            map.insert("trained".into(), one);
            return Ok(map);
        }
    }
    if let Ok(map) = serde_json::from_str::<BTreeMap<String, ScoreResult>>(&text) {
        return Ok(map);
    }
    Err(format!("could not parse scorer stdout: {text}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_map_stdout() {
        let raw = br#"{"trained":{"score":0.5,"error":0.25,"complexityPenalty":0.01}}"#;
        let map = parse_scorer_stdout(raw).unwrap();
        assert!((map["trained"].score - 0.5).abs() < 1e-15);
    }

    #[test]
    fn an_empty_batch_is_refused() {
        let err = score_creatures(
            Path::new("/nonexistent-scorer"),
            &[],
            Path::new("data"),
            Path::new("work"),
        )
        .unwrap_err();
        assert!(err.contains("no creatures to score"), "{err}");
    }

    #[cfg(unix)]
    mod batch {
        use super::*;
        use std::fs;
        use std::os::unix::fs::PermissionsExt;
        use tempfile::tempdir;

        /// A stub `rust_scorer` printing `body` — the map a real batch call
        /// returns, keyed by file stem.
        fn stub(dir: &Path, body: &str) -> std::path::PathBuf {
            let path = dir.join("stub-scorer");
            fs::write(&path, format!("#!/bin/sh\nset -eu\nprintf '{body}\\n'\n")).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
            path
        }

        /// Results come back in the order the candidates were requested, not in
        /// the order the scorer happened to print them — the mapping is by
        /// stem, so a lost candidate can never be charged to its neighbour.
        #[test]
        fn results_follow_the_request_order_not_the_scorer_order() {
            let dir = tempdir().unwrap();
            let scorer = stub(
                dir.path(),
                r#"{"rung-0":{"score":0.1,"error":0.0},"rung-1":{"score":0.9,"error":0.0}}"#,
            );
            let scored = score_creatures(
                &scorer,
                &[("rung-1", "{}"), ("rung-0", "{}")],
                dir.path(),
                &dir.path().join("work"),
            )
            .unwrap();
            assert_eq!(
                scored.iter().map(|s| s.score).collect::<Vec<_>>(),
                vec![0.9, 0.1]
            );
            // Every candidate reached the scorer as its own file.
            let written = dir.path().join("work").join("scorer-candidate");
            assert!(written.join("rung-0.json").is_file());
            assert!(written.join("rung-1.json").is_file());
        }

        /// A candidate the scorer did not report fails loudly rather than
        /// shifting every later result onto the wrong rung.
        #[test]
        fn a_dropped_candidate_fails_loudly() {
            let dir = tempdir().unwrap();
            let scorer = stub(dir.path(), r#"{"rung-0":{"score":0.1,"error":0.0}}"#);
            let err = score_creatures(
                &scorer,
                &[("rung-0", "{}"), ("rung-1", "{}")],
                dir.path(),
                &dir.path().join("work"),
            )
            .unwrap_err();
            assert!(err.contains("no score for candidate 'rung-1'"), "{err}");
        }

        /// A single candidate is matched by position, deliberately: callers
        /// have always accepted whatever stem the scorer chose to echo, and
        /// tightening that would change the untouched baseline / line-search
        /// path. Pinned so the leniency is a decision, not an accident — with
        /// two candidates the same response fails loudly (test above).
        #[test]
        fn a_single_candidate_accepts_whatever_stem_the_scorer_echoes() {
            let dir = tempdir().unwrap();
            let scorer = stub(
                dir.path(),
                r#"{"something-else":{"score":0.7,"error":0.0}}"#,
            );
            let scored = score_creature(&scorer, "{}", dir.path(), &dir.path().join("work"))
                .expect("single candidate is matched by position");
            assert!((scored.score - 0.7).abs() < 1e-15);
        }

        /// The candidate directory is reused across epochs, so a stale file
        /// from a longer earlier ladder must not be scored again.
        #[test]
        fn the_candidate_directory_is_cleared_between_batches() {
            let dir = tempdir().unwrap();
            let scorer = stub(dir.path(), r#"{"rung-0":{"score":0.1,"error":0.0}}"#);
            let work = dir.path().join("work");
            let stale = work.join("scorer-candidate");
            fs::create_dir_all(&stale).unwrap();
            fs::write(stale.join("rung-9.json"), "{}").unwrap();
            score_creatures(&scorer, &[("rung-0", "{}")], dir.path(), &work).unwrap();
            assert!(!stale.join("rung-9.json").exists());
        }
    }
}
