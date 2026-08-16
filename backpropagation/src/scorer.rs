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

/// Score a single creature JSON by writing it into a temp directory and
/// invoking `rust_scorer <dir> <training-data>`.
pub fn score_creature(
    scorer: &Path,
    creature_json: &str,
    training_data: &Path,
    work_dir: &Path,
) -> Result<ScoreResult, String> {
    let candidates = work_dir.join("scorer-candidate");
    fs::create_dir_all(&candidates).map_err(|e| e.to_string())?;
    let path = candidates.join("trained.json");
    fs::write(&path, creature_json).map_err(|e| e.to_string())?;
    let output = Command::new(scorer)
        .arg(&candidates)
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
    let scores = parse_scorer_stdout(&output.stdout)?;
    scores
        .into_values()
        .next()
        .ok_or_else(|| "scorer returned no creature scores".into())
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
}
