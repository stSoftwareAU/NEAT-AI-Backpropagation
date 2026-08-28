//! Creature JSON tags (`{ name, value }[]`) for GRQ check-in compatibility.
//!
//! `neat_core::CreatureExport` does not round-trip `tags`, so this crate keeps
//! them in [`CreatureMeta`] and re-attaches on write — preserving pedigree tags
//! while stamping `score` / `error` / `backpropagation` for GRQ
//! (`worker/Backprop/run.sh` reads those tags; see GRQ #3991 / #3952).
//!
//! That applies to **per-neuron** tags too (GRQ #4491). `NeuronExport` models
//! no `tags` field either, so every neuron's discovery / intelligent-design
//! provenance was dropped on write: all seven `*-backprop.json` samples in
//! GRQ-sampler carry 0 tagged neurons where the champions they descend from
//! carry ~2,500. GRQ's #4216 check-in guard refuses a candidate that lost the
//! source's per-neuron tags, so once that guard was wired into the Backprop
//! worker (GRQ #4318) no Backprop run could publish at all — the failure
//! surfaced as a rebase problem (GRQ #4491) because a rebase faithfully
//! carries forward the empty tag set it was handed.
//!
//! Training moves biases and weights and never genes (issue #94), so every
//! source neuron uuid survives; the sidecar is still reconciled against the
//! neurons actually written, so a tag can never name a neuron that is not
//! there.
//!
//! The creature-level `uuid` is deliberately *not* kept (issue #101). It is a
//! content-derived v5 hash over the creature's neurons, synapses and `input`,
//! and training moves every bias and weight — so re-attaching the source uuid
//! publishes a hash that no longer describes the content. Tags are excluded
//! from that hash, which is why they are safe to carry across. Emitting no
//! uuid lets the consumer derive it from the content it actually received.

use std::collections::BTreeMap;

use crate::creature_io::ObservationWidth;
use neat_core::CreatureExport;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One creature tag (NEAT-AI / `@stsoftware/tags` shape).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreatureTag {
    /// Tag key (`score`, `error`, `name`, `backpropagation`, …).
    pub name: String,
    /// Tag value (always a string in the export format).
    pub value: String,
}

/// Top-level fields stripped by `parse_creature_json` that we must keep.
///
/// The source `uuid` is not among them (issue #101) — see the module docs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CreatureMeta {
    /// Ordered tags (upserts replace by name, preserving order of first insert).
    pub tags: Vec<CreatureTag>,
    /// Per-neuron tags keyed by the neuron's `uuid` (GRQ #4491). Only neurons
    /// that carried at least one tag appear here.
    pub neuron_tags: BTreeMap<String, Vec<CreatureTag>>,
}

fn parse_tag_array(value: Option<&Value>) -> Vec<CreatureTag> {
    value
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
        .unwrap_or_default()
}

impl CreatureMeta {
    /// Parse creature-level and per-neuron `tags` from raw creature JSON
    /// (missing → empty).
    pub fn from_creature_json(text: &str) -> Self {
        let Ok(value) = serde_json::from_str::<Value>(text) else {
            return Self::default();
        };
        let tags = parse_tag_array(value.get("tags"));
        let mut neuron_tags = BTreeMap::new();
        if let Some(neurons) = value.get("neurons").and_then(|v| v.as_array()) {
            for neuron in neurons {
                let Some(uuid) = neuron.get("uuid").and_then(|u| u.as_str()) else {
                    continue;
                };
                let tags = parse_tag_array(neuron.get("tags"));
                if !tags.is_empty() {
                    neuron_tags.insert(uuid.to_string(), tags);
                }
            }
        }
        Self { tags, neuron_tags }
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
    source_width: ObservationWidth,
) -> Result<Value, String> {
    // Issue #92: never write a widthless creature. `checked_json` asserts the
    // struct and the serialised bytes both carry the source `input` /
    // `output` (each ≥ 1) before tags are re-attached.
    let body = source_width.checked_json(creature)?;
    let mut value: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    // No creature-level `uuid` is re-attached (issue #101): it is a content
    // hash and training has moved the content it hashes over. Tags are
    // excluded from that hash, so they are safe to carry.
    if !meta.tags.is_empty() {
        value["tags"] = serde_json::to_value(&meta.tags).map_err(|e| e.to_string())?;
    }
    // Per-neuron provenance (GRQ #4491). Keyed by the neuron uuid the source
    // carried, which survives training, and reconciled against the neurons
    // actually written — a sidecar entry naming a neuron that is not in the
    // output is dropped rather than inventing one.
    if !meta.neuron_tags.is_empty()
        && let Some(neurons) = value.get_mut("neurons").and_then(|n| n.as_array_mut())
    {
        for neuron in neurons.iter_mut() {
            let Some(uuid) = neuron.get("uuid").and_then(|u| u.as_str()) else {
                continue;
            };
            let Some(tags) = meta.neuron_tags.get(uuid) else {
                continue;
            };
            let tags = serde_json::to_value(tags).map_err(|e| e.to_string())?;
            neuron["tags"] = tags;
        }
    }
    Ok(value)
}

/// Pretty-print a creature with `tags` re-attached for check-in.
///
/// No creature-level `uuid` is emitted (issue #101) — the consumer derives it
/// from the content it receives.
///
/// `source_width` is the observation width of the creature the run started
/// from ([`ObservationWidth::of`]); the call fails — and nothing should be
/// written — when `creature` does not carry exactly that width, or when the
/// serialised text would not (issue #92).
pub fn serialize_creature_with_meta(
    creature: &CreatureExport,
    meta: &CreatureMeta,
    source_width: ObservationWidth,
) -> Result<String, String> {
    let value = creature_value_with_meta(creature, meta, source_width)?;
    let mut out = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    if !out.ends_with('\n') {
        out.push('\n');
    }
    // Belt and braces: the exact bytes handed to `best.json` still carry the
    // width after `tags` were re-attached.
    source_width.assert_written(&out)?;
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
      "neurons": [{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY",
                   "tags":[{"name":"discovery","value":"🔬 scan 42"}]}],
      "synapses": [{"fromUUID":"input-0","toUUID":"o1","weight":1.0}],
      "tags": [
        {"name":"name","value":"Tiny"},
        {"name":"score","value":"0.1"},
        {"name":"lamarck","value":"🦒 leave me alone"}
      ]
    }"#;

    /// Issue #101: the uuid half of this assertion encoded the defect — the
    /// creature-level uuid is content-derived, so this crate never keeps it.
    #[test]
    fn extract_preserves_tags() {
        let meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        assert_eq!(meta.tags[0].name, "name");
        assert_eq!(meta.tags[0].value, "Tiny");
    }

    /// Issue #101: the check-in text never carries a creature-level `uuid`,
    /// even when the source did — the consumer derives it from the content it
    /// actually received.
    #[test]
    fn serialize_never_emits_a_creature_level_uuid() {
        let creature = parse_creature_json(TINY_TAGGED).unwrap();
        let meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        let width = ObservationWidth::of(&creature).unwrap();
        let text = serialize_creature_with_meta(&creature, &meta, width).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert!(value.get("uuid").is_none(), "unexpected uuid in {text}");
        // Per-neuron uuid is a different concept — a stable identity label
        // that is an *input* to the creature hash — and must survive.
        assert_eq!(value["neurons"][0]["uuid"], "o1");
        assert_eq!(value["tags"][0]["name"], "name");
    }

    /// GRQ #4491: per-neuron provenance is parsed into the sidecar and
    /// re-attached by uuid on write.
    #[test]
    fn per_neuron_tags_round_trip_by_uuid() {
        let creature = parse_creature_json(TINY_TAGGED).unwrap();
        let meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        assert_eq!(meta.neuron_tags.len(), 1, "{:?}", meta.neuron_tags);
        assert_eq!(meta.neuron_tags["o1"][0].name, "discovery");

        let width = ObservationWidth::of(&creature).unwrap();
        let text = serialize_creature_with_meta(&creature, &meta, width).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["neurons"][0]["tags"][0]["name"], "discovery");
        assert_eq!(value["neurons"][0]["tags"][0]["value"], "🔬 scan 42");
    }

    /// A sidecar entry naming a neuron the written creature does not have is
    /// dropped — the writer never invents a neuron to hang provenance on.
    #[test]
    fn a_tag_for_an_absent_neuron_is_not_written() {
        let creature = parse_creature_json(TINY_TAGGED).unwrap();
        let mut meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        meta.neuron_tags.insert(
            "ghost".to_string(),
            vec![CreatureTag {
                name: "discovery".to_string(),
                value: "👻".to_string(),
            }],
        );
        let width = ObservationWidth::of(&creature).unwrap();
        let text = serialize_creature_with_meta(&creature, &meta, width).unwrap();
        assert!(!text.contains("👻"), "{text}");
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["neurons"].as_array().unwrap().len(), 1);
    }

    /// A creature whose neurons carry no tags gains none.
    #[test]
    fn an_untagged_neuron_stays_untagged() {
        let untagged = r#"{
          "input": 1,
          "output": 1,
          "forwardOnly": true,
          "neurons": [{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
          "synapses": [{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
        }"#;
        let creature = parse_creature_json(untagged).unwrap();
        let meta = CreatureMeta::from_creature_json(untagged);
        assert!(meta.neuron_tags.is_empty());
        let width = ObservationWidth::of(&creature).unwrap();
        let text = serialize_creature_with_meta(&creature, &meta, width).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert!(value["neurons"][0].get("tags").is_none(), "{text}");
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
        let width = ObservationWidth::of(&creature).unwrap();
        let text = serialize_creature_with_meta(&creature, &meta, width).unwrap();
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

    /// Issue #92: the check-in serialiser refuses a creature whose width no
    /// longer matches the source, and one whose width is < 1.
    #[test]
    fn serialize_rejects_a_mismatched_observation_width() {
        let creature = parse_creature_json(TINY_TAGGED).unwrap();
        let meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        let wider = ObservationWidth {
            input: 2,
            output: 1,
        };
        let err = serialize_creature_with_meta(&creature, &meta, wider)
            .expect_err("input 1 vs source 2 must be refused");
        assert!(err.contains("observation width changed"), "{err}");
        assert!(err.contains("source input=2 output=1"), "{err}");

        let mut drifted = creature.clone();
        drifted.output = 3;
        let width = ObservationWidth::of(&creature).unwrap();
        let err = serialize_creature_with_meta(&drifted, &meta, width)
            .expect_err("output 3 vs source 1 must be refused");
        assert!(err.contains("written input=1 output=3"), "{err}");
    }

    #[test]
    fn serialize_rejects_a_widthless_creature() {
        let mut creature = parse_creature_json(TINY_TAGGED).unwrap();
        let meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        creature.input = 0;
        let zero = ObservationWidth {
            input: 0,
            output: 1,
        };
        let err = serialize_creature_with_meta(&creature, &meta, zero)
            .expect_err("input 0 must never be written");
        assert_eq!(err, "Must have at least one input neurons was: 0");
    }

    /// Issue #92: a valid source round-trips `input` / `output` into the
    /// check-in text as the same integers.
    #[test]
    fn serialize_preserves_the_source_observation_width() {
        let creature = parse_creature_json(TINY_TAGGED).unwrap();
        let meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        let width = ObservationWidth::of(&creature).unwrap();
        let text = serialize_creature_with_meta(&creature, &meta, width).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["input"], 1);
        assert_eq!(value["output"], 1);
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
