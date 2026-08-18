//! Shared creature loading for the subcommands that drive the accumulate
//! engine (issue #54).
//!
//! Every entry point encodes the same rule: read the JSON, parse it, reject a
//! creature without an observation width, and reject anything that is not
//! forward-only. Keeping the sequence here means the supported-graph rule —
//! and its wording — has a single owner.
//!
//! The observation width (issue #92) is the pair of top-level `input` /
//! `output` integers on the export. `neurons` lists only the *non-input*
//! neurons, so `input` cannot be re-derived from the graph — a creature that
//! arrives with `input < 1` or `output < 1` is rejected here, before any
//! epoch runs, and [`ObservationWidth::assert_written`] refuses to hand back a
//! check-in creature that lost or changed the width. This is a local guard at
//! the binary boundary; it stays even once `neat-core` validates the same
//! rule (NEAT-AI-core#550).

use neat_core::{CreatureExport, creature_to_json, creature_to_json_pretty, parse_creature_json};
use serde_json::Value;
use std::fs;
use std::path::Path;

/// Error returned for a creature this trainer cannot backpropagate.
pub const FORWARD_ONLY_REQUIRED: &str =
    "this trainer supports forward-only creatures only (no re-entrant / recurrent graphs)";

/// The authoritative observation width of a creature: the top-level `input`
/// / `output` counts (issue #92).
///
/// Built from the *source* creature once and threaded to every write, so a
/// derived creature (post-apply candidate, `best.json`) is checked against
/// the width the run started from rather than against itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservationWidth {
    /// Top-level `input` count of the source creature.
    pub input: usize,
    /// Top-level `output` count of the source creature.
    pub output: usize,
}

impl ObservationWidth {
    /// Capture the width of `creature`, rejecting `input < 1` / `output < 1`.
    ///
    /// Wording mirrors NEAT-AI `CreatureValidate.ts` so a GRQ log line reads
    /// the same whichever side caught it.
    pub fn of(creature: &CreatureExport) -> Result<Self, String> {
        check_observation_width(creature.input, creature.output)?;
        Ok(Self {
            input: creature.input,
            output: creature.output,
        })
    }

    /// Assert `creature` still carries exactly this width (and that the width
    /// is `>= 1`).
    ///
    /// Call before serialising any creature that will be written to disk or
    /// handed back over the FFI — never write a widthless creature.
    pub fn assert_matches(self, creature: &CreatureExport) -> Result<(), String> {
        check_observation_width(self.input, self.output)?;
        if creature.input != self.input || creature.output != self.output {
            return Err(format!(
                "refusing to write creature: observation width changed \
                 (source input={} output={}, written input={} output={})",
                self.input, self.output, creature.input, creature.output
            ));
        }
        Ok(())
    }

    /// Assert the serialised JSON `text` carries this width as top-level
    /// integers — the bytes that land in `best.json`, not the in-memory
    /// struct.
    pub fn assert_written(self, text: &str) -> Result<(), String> {
        check_observation_width(self.input, self.output)?;
        let value: Value = serde_json::from_str(text)
            .map_err(|e| format!("refusing to write creature: not JSON: {e}"))?;
        let read = |key: &str| -> Result<usize, String> {
            value
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    format!(
                        "refusing to write creature: top-level `{key}` is missing or not a \
                         non-negative integer"
                    )
                })
        };
        let (input, output) = (read("input")?, read("output")?);
        if input != self.input || output != self.output {
            return Err(format!(
                "refusing to write creature: serialised observation width changed \
                 (source input={} output={}, serialised input={input} output={output})",
                self.input, self.output
            ));
        }
        Ok(())
    }

    /// Serialise `creature` compactly, refusing when its width is not
    /// this one. Use for every on-disk / FFI creature except the tagged
    /// check-in file (see [`crate::tags::serialize_creature_with_meta`]).
    pub fn checked_json(self, creature: &CreatureExport) -> Result<String, String> {
        self.assert_matches(creature)?;
        let text = creature_to_json(creature).map_err(|e| e.to_string())?;
        self.assert_written(&text)?;
        Ok(text)
    }

    /// Pretty-printed [`ObservationWidth::checked_json`].
    pub fn checked_json_pretty(self, creature: &CreatureExport) -> Result<String, String> {
        self.assert_matches(creature)?;
        let text = creature_to_json_pretty(creature).map_err(|e| e.to_string())?;
        self.assert_written(&text)?;
        Ok(text)
    }
}

/// Reject `input < 1` / `output < 1` (issue #92; wording as in NEAT-AI
/// `CreatureValidate.ts`).
pub fn check_observation_width(input: usize, output: usize) -> Result<(), String> {
    if input < 1 {
        return Err(format!("Must have at least one input neurons was: {input}"));
    }
    if output < 1 {
        return Err(format!(
            "Must have at least one output neurons was: {output}"
        ));
    }
    Ok(())
}

/// Parse creature JSON text, rejecting a missing observation width and
/// non-forward-only graphs.
///
/// Use this when the caller already holds the file text (`train` also parses
/// it for [`crate::tags::CreatureMeta`]); otherwise prefer
/// [`load_forward_only_creature`].
pub fn parse_forward_only_creature(text: &str) -> Result<CreatureExport, String> {
    let creature = parse_creature_json(text).map_err(|e| e.to_string())?;
    check_observation_width(creature.input, creature.output)?;
    if !creature.forward_only {
        return Err(FORWARD_ONLY_REQUIRED.into());
    }
    Ok(creature)
}

/// Read `path` and parse it as a forward-only creature.
///
/// Fails loudly on a missing/unreadable file, malformed JSON, or a re-entrant
/// graph — never returns a partially usable creature.
pub fn load_forward_only_creature(path: &Path) -> Result<CreatureExport, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    parse_forward_only_creature(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECURRENT: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":false,"input":1,"output":1,
      "neurons":[{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
      "synapses":[{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
    }"#;

    const FORWARD_ONLY: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
      "neurons":[{"type":"output","uuid":"o1","bias":0.25,"squash":"IDENTITY"}],
      "synapses":[{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
    }"#;

    #[test]
    fn parse_rejects_a_recurrent_creature() {
        let err = parse_forward_only_creature(RECURRENT).expect_err("recurrent must be rejected");
        assert_eq!(err, FORWARD_ONLY_REQUIRED);
    }

    #[test]
    fn parse_accepts_a_forward_only_creature() {
        let creature = parse_forward_only_creature(FORWARD_ONLY).unwrap();
        assert!(creature.forward_only);
        assert_eq!(creature.neurons[0].bias, 0.25);
    }

    #[test]
    fn parse_reports_malformed_json() {
        let err =
            parse_forward_only_creature("{ not json").expect_err("malformed must be rejected");
        assert_ne!(err, FORWARD_ONLY_REQUIRED);
        assert!(!err.is_empty());
    }
}
