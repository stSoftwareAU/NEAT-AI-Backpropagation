//! Creature JSON tags (`{ name, value }[]`) for GRQ check-in compatibility.
//!
//! `neat_core::CreatureExport` does not round-trip `tags` / `uuid`, so this
//! crate keeps them in [`CreatureMeta`] and re-attaches on write — preserving
//! pedigree tags while stamping `score` / `error` / `backpropagation` for GRQ
//! (`worker/Backprop/run.sh` reads those tags; see GRQ #3991 / #3952).

use neat_core::{CreatureExport, creature_to_json};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// One creature tag (NEAT-AI / `@stsoftware/tags` shape).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreatureTag {
    /// Tag key (`score`, `error`, `name`, `backpropagation`, …).
    pub name: String,
    /// Tag value (always a string in the export format).
    pub value: String,
}

/// Top-level fields stripped by `parse_creature_json` that we must keep.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CreatureMeta {
    /// Optional creature UUID from the source JSON.
    pub uuid: Option<String>,
    /// Ordered tags (upserts replace by name, preserving order of first insert).
    pub tags: Vec<CreatureTag>,
}

impl CreatureMeta {
    /// Parse `uuid` + `tags` from raw creature JSON (missing → empty).
    pub fn from_creature_json(text: &str) -> Self {
        let Ok(value) = serde_json::from_str::<Value>(text) else {
            return Self::default();
        };
        let uuid = value
            .get("uuid")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let tags = value
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        let name = t.get("name")?.as_str()?.to_string();
                        let value = match t.get("value")? {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        Some(CreatureTag { name, value })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self { uuid, tags }
    }

    /// Insert or replace a tag by name.
    pub fn upsert(&mut self, name: &str, value: impl Into<String>) {
        let value = value.into();
        if let Some(existing) = self.tags.iter_mut().find(|t| t.name == name) {
            existing.value = value;
        } else {
            self.tags.push(CreatureTag {
                name: name.to_string(),
                value,
            });
        }
    }

    /// Update score/error and stamp a run-level backpropagation summary.
    ///
    /// Only the `backpropagation` tag is ours. Never touch `lamarck` or
    /// `intelligentDesign` (GRQ #3952 — different programs).
    pub fn stamp_train_result(&mut self, progress: &BackpropProgress) {
        self.upsert("score", format!("{}", progress.score));
        self.upsert("error", format!("{}", progress.error));
        self.upsert("backpropagation", backprop_progress_message(progress));
    }
}

/// Fields for the run-level backpropagation check-in summary.
#[derive(Debug, Clone, Copy)]
pub struct BackpropProgress {
    /// Epochs that accepted an apply.
    pub accepted_epochs: u64,
    /// Epochs requested.
    pub epochs: u64,
    /// Authoritative `rust_scorer` score of the written creature.
    pub score: f64,
    /// Authoritative error of the written creature.
    pub error: f64,
    /// Baseline score before the run (for cumulative Δ wording).
    pub opening_score: f64,
}

/// Run-level check-in blurb for GRQ commit subjects, marked 🌀 (issue #31).
pub fn backprop_progress_message(progress: &BackpropProgress) -> String {
    let score_clause = format_score_improved(progress.score, progress.opening_score);
    let accept_word = if progress.accepted_epochs == 1 {
        "accept"
    } else {
        "accepts"
    };
    let epoch_word = if progress.epochs == 1 {
        "epoch"
    } else {
        "epochs"
    };
    format!(
        "🌀 · {} {accept_word} / {} {epoch_word} · {score_clause}",
        progress.accepted_epochs, progress.epochs,
    )
}

fn format_score_improved(score: f64, opening: f64) -> String {
    let formatted = format_g(score, 6);
    let delta = score - opening;
    if delta < 0.0 {
        format!("score: {formatted} declined by {}", format_g(-delta, 3))
    } else {
        format!("score: {formatted} improved by {}", format_g(delta, 3))
    }
}

fn format_g(v: f64, prec: usize) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let prec = prec.max(1);
    let abs = v.abs();
    let exp = abs.log10().floor() as i32;
    if exp < -4 || exp >= prec as i32 {
        let digits = prec.saturating_sub(1);
        let s = format!("{v:.digits$e}");
        return trim_g_scientific(&s);
    }
    let decimals = (prec as i32 - exp - 1).max(0) as usize;
    let s = format!("{v:.decimals$}");
    trim_trailing_zeros_and_dot(&s)
}

fn trim_g_scientific(s: &str) -> String {
    let Some((mant, exp)) = s.split_once('e') else {
        return s.to_string();
    };
    let mant = trim_trailing_zeros_and_dot(mant);
    let exp_i: i32 = exp.parse().unwrap_or(0);
    format!("{mant}e{exp_i:+03}")
}

fn trim_trailing_zeros_and_dot(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let mut out = s.to_string();
    while out.ends_with('0') {
        out.pop();
    }
    if out.ends_with('.') {
        out.pop();
    }
    out
}

fn creature_value_with_meta(
    creature: &CreatureExport,
    meta: &CreatureMeta,
) -> Result<Value, String> {
    let body = creature_to_json(creature).map_err(|e| e.to_string())?;
    let mut value: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    if let Some(uuid) = &meta.uuid {
        value["uuid"] = json!(uuid);
    }
    if !meta.tags.is_empty() {
        value["tags"] = serde_json::to_value(&meta.tags).map_err(|e| e.to_string())?;
    }
    Ok(value)
}

/// Pretty-print a creature with `uuid` / `tags` re-attached for check-in.
pub fn serialize_creature_with_meta(
    creature: &CreatureExport,
    meta: &CreatureMeta,
) -> Result<String, String> {
    let value = creature_value_with_meta(creature, meta)?;
    let mut out = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    if !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;

    const TINY_TAGGED: &str = r#"{
      "uuid": "creature-1",
      "input": 1,
      "output": 1,
      "forwardOnly": true,
      "neurons": [{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
      "synapses": [{"fromUUID":"input-0","toUUID":"o1","weight":1.0}],
      "tags": [
        {"name":"name","value":"Tiny"},
        {"name":"score","value":"0.1"},
        {"name":"lamarck","value":"🦒 leave me alone"}
      ]
    }"#;

    #[test]
    fn extract_preserves_uuid_and_tags() {
        let meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        assert_eq!(meta.uuid.as_deref(), Some("creature-1"));
        assert_eq!(meta.tags[0].name, "name");
        assert_eq!(meta.tags[0].value, "Tiny");
    }

    #[test]
    fn stamp_updates_score_and_backpropagation_without_touching_lamarck() {
        let creature = parse_creature_json(TINY_TAGGED).unwrap();
        let mut meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        meta.stamp_train_result(&BackpropProgress {
            accepted_epochs: 2,
            epochs: 4,
            score: 0.35,
            error: 0.65,
            opening_score: 0.34,
        });
        let text = serialize_creature_with_meta(&creature, &meta).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        let tags = value["tags"].as_array().unwrap();
        let score = tags.iter().find(|t| t["name"] == "score").unwrap();
        assert_eq!(score["value"], "0.35");
        let backprop = tags
            .iter()
            .find(|t| t["name"] == "backpropagation")
            .unwrap();
        let msg = backprop["value"].as_str().unwrap();
        assert!(msg.starts_with("🌀 · "), "unexpected subject: {msg}");
        assert!(!msg.contains('🔁'));
        assert!(!msg.contains("Backprop"));
        assert!(msg.contains("2 accepts / 4 epochs"));
        assert!(msg.contains("improved by"));
        let lamarck = tags.iter().find(|t| t["name"] == "lamarck").unwrap();
        assert_eq!(lamarck["value"], "🦒 leave me alone");
    }

    #[test]
    fn progress_message_is_spiral_prefixed_with_singular_wording() {
        let msg = backprop_progress_message(&BackpropProgress {
            accepted_epochs: 1,
            epochs: 1,
            score: 0.5,
            error: 0.5,
            opening_score: 0.4,
        });
        assert_eq!(msg, "🌀 · 1 accept / 1 epoch · score: 0.5 improved by 0.1");
    }

    #[test]
    fn progress_message_reports_a_decline() {
        let msg = backprop_progress_message(&BackpropProgress {
            accepted_epochs: 0,
            epochs: 3,
            score: 0.4,
            error: 0.6,
            opening_score: 0.5,
        });
        assert_eq!(
            msg,
            "🌀 · 0 accepts / 3 epochs · score: 0.4 declined by 0.1"
        );
    }
}
