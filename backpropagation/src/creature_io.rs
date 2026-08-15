//! Shared creature loading for the subcommands that drive the accumulate
//! engine (issue #54).
//!
//! Every entry point encodes the same rule: read the JSON, parse it, and
//! reject anything that is not forward-only. Keeping the sequence here means
//! the supported-graph rule — and its wording — has a single owner.

use neat_core::{CreatureExport, parse_creature_json};
use std::fs;
use std::path::Path;

/// Error returned for a creature this trainer cannot backpropagate.
pub const FORWARD_ONLY_REQUIRED: &str =
    "this trainer supports forward-only creatures only (no re-entrant / recurrent graphs)";

/// Parse creature JSON text, rejecting non-forward-only graphs.
///
/// Use this when the caller already holds the file text (`train` also parses
/// it for [`crate::tags::CreatureMeta`]); otherwise prefer
/// [`load_forward_only_creature`].
pub fn parse_forward_only_creature(text: &str) -> Result<CreatureExport, String> {
    let creature = parse_creature_json(text).map_err(|e| e.to_string())?;
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
