//! Output gate (issue #94): no trained creature leaves this crate without
//! `neat_core::creature_validate` certifying it.
//!
//! Backpropagation rewrites biases and weights, so it is the consumer most
//! likely to produce the *numeric* failures the shared validator checks — a
//! diverged epoch that writes a `NaN` bias currently escapes as a "trained"
//! creature and only breaks much later, in whatever loads it. The gate runs
//! **at output**: after a run has finished and before the creature is scored,
//! written or returned. Loading is deliberately not gated — an
//! externally-supplied creature is not this crate's bug to report.
//!
//! The check itself lives in `neat-core`, which is the single definition of a
//! valid creature (NEAT-AI#3800). Nothing here re-implements a rule.

use neat_core::{CreatureExport, ValidateOptions, ValidationStats, creature_validate};

/// The topology a training run must hand back unchanged.
///
/// Captured from the *source* creature once, exactly as
/// [`crate::creature_io::ObservationWidth`] pins the observation width, then
/// checked against every creature the run produces. Backpropagation moves
/// values, never structure, so a neuron or synapse count that changed during
/// training is a defect in the run — pinning the counts turns
/// [`neat_core::creature_validate()`] into a check that training preserved the
/// topology as well as a check that the values are sane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainedTopology {
    /// Total neuron count, `input + neurons.len()` — the total
    /// `creature_validate` compares against, since the export form lists only
    /// the non-input neurons.
    pub neurons: usize,
    /// Synapse count.
    pub connections: usize,
}

impl TrainedTopology {
    /// Pin the topology of the creature a run starts from.
    pub fn of(source: &CreatureExport) -> Self {
        Self {
            neurons: source.input + source.neurons.len(),
            connections: source.synapses.len(),
        }
    }

    /// The options this crate validates its output with.
    ///
    /// Deliberate choices, one per field:
    ///
    /// - `neurons` / `connections` — pinned from the source creature (see
    ///   [`TrainedTopology`]). Training must not add or drop a gene, so the
    ///   expected counts are always available and always worth checking.
    /// - `forward_only` — `true`. Every creature this trainer accepts is
    ///   forward-only ([`crate::creature_io::FORWARD_ONLY_REQUIRED`]), so its
    ///   output is held to the same feed-forward contract it was loaded
    ///   under: sorted, self-loop free and acyclic.
    /// - `feedback_loop` — left `None`. `forward_only` already resolves it to
    ///   `Some(false)`; setting it as well would only invite the two to drift
    ///   apart.
    pub fn options(self) -> ValidateOptions {
        ValidateOptions {
            neurons: Some(self.neurons),
            connections: Some(self.connections),
            feedback_loop: None,
            forward_only: true,
        }
    }

    /// Validate a creature this crate produced, naming `produced_by` when it
    /// is rejected.
    ///
    /// `produced_by` identifies the run — and, where a run yields more than
    /// one creature, which one — so a violation is attributed to the training
    /// that produced it rather than to whatever later chokes on the file.
    ///
    /// # Errors
    ///
    /// Returns the [`neat_core::ValidationFailure`]'s class, reason, message
    /// and offending neuron / synapse index as a single loud string. The
    /// caller must abandon the creature: it is never scored, written or
    /// returned.
    pub fn assert_valid(
        self,
        creature: &CreatureExport,
        produced_by: &str,
    ) -> Result<ValidationStats, String> {
        creature_validate(creature, &self.options()).map_err(|failure| {
            let at = match (failure.neuron_index, failure.synapse_index) {
                (Some(neuron), _) => format!(" [neuron index {neuron}]"),
                (None, Some(synapse)) => format!(" [synapse index {synapse}]"),
                (None, None) => String::new(),
            };
            format!(
                "refusing to return a trained creature: {produced_by} produced a creature \
                 neat-core rejected — {failure}{at}"
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;

    /// Identity chain `input-0 → h1 → o1`.
    const VALID: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
      "neurons":[
        {"type":"hidden","uuid":"h1","bias":0.25,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":-0.5,"squash":"IDENTITY"}
      ],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0}
      ]
    }"#;

    fn valid() -> CreatureExport {
        parse_creature_json(VALID).unwrap()
    }

    #[test]
    fn pins_the_source_topology() {
        assert_eq!(
            TrainedTopology::of(&valid()),
            TrainedTopology {
                neurons: 3,
                connections: 2
            }
        );
    }

    #[test]
    fn options_pin_the_counts_and_demand_forward_only() {
        let options = TrainedTopology::of(&valid()).options();
        assert_eq!(options.expected_neurons(), Some(3));
        assert_eq!(options.expected_connections(), Some(2));
        assert!(options.forward_only);
        // `forwardOnly` resolves the feedback-loop flag on its own.
        assert_eq!(options.resolved_feedback_loop(), Some(false));
        assert!(options.rejects_recursive_synapses());
    }

    #[test]
    fn a_healthy_trained_creature_passes() {
        let source = valid();
        let mut trained = source.clone();
        // What training does: move values, leave the graph alone.
        trained.neurons[0].bias = 0.3125;
        trained.synapses[1].weight = 0.75;
        let stats = TrainedTopology::of(&source)
            .assert_valid(&trained, "unit test")
            .expect("a value-only change must stay valid");
        assert_eq!(stats.neurons(), 3);
    }

    #[test]
    fn a_non_finite_bias_is_reported_with_its_neuron_index() {
        let source = valid();
        let mut diverged = source.clone();
        diverged.neurons[0].bias = f64::NAN;
        let err = TrainedTopology::of(&source)
            .assert_valid(&diverged, "epoch 7 of the unit-test run")
            .expect_err("a NaN bias must not be certified");
        assert!(
            err.contains("refusing to return a trained creature"),
            "{err}"
        );
        assert!(err.contains("epoch 7 of the unit-test run"), "{err}");
        // Class, reason and message come from neat-core verbatim.
        assert!(err.contains("Validation"), "{err}");
        assert!(err.contains("bias"), "{err}");
        // The hidden neuron sits at index `input + 0`.
        assert!(err.contains("[neuron index 1]"), "{err}");
    }

    #[test]
    fn a_dropped_neuron_is_rejected_by_the_pinned_count() {
        let source = valid();
        let mut pruned = source.clone();
        pruned.neurons.remove(0);
        pruned.synapses.remove(0);
        let err = TrainedTopology::of(&source)
            .assert_valid(&pruned, "unit test")
            .expect_err("training must not change the topology");
        assert!(err.contains("Neurons length: 2 expected: 3"), "{err}");
    }
}
