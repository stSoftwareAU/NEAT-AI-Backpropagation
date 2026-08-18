//! NEAT-AI `CreatureTrace` wire format for `train` artifacts (issue #78).
//!
//! NEAT-AI's TypeScript trainer writes `creature.traceJSON()` into
//! `TrainOptions.traceStore` when an iteration makes the network worse, and
//! keeps the best iteration's trace on `TrainingResult.trace`. A `CreatureTrace`
//! is the ordinary UUID-only [`CreatureExport`] with a `trace` object added to
//! every neuron and synapse the pass actually accumulated.
//!
//! This module serialises the Rust accumulate pass into that same shape, so a
//! trace written by `train` loads in NEAT-AI without an adapter:
//!
//! * the export half is [`neat_core`]'s own serialisation — no field of the
//!   creature format is restated here;
//! * neuron `trace` mirrors NEAT-AI `NeuronState`, synapse `trace` mirrors
//!   `SynapseState`;
//! * a gene with a zero accumulation count carries no `trace` key at all,
//!   matching `traceJSON()`'s `if (state.count)` guard.

use crate::backprop::LearningSignal;
use crate::propagate_layout::{AccumulateReport, NeuronTraceStats};
use neat_core::CreatureExport;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::Path;

/// Per-neuron trace state — NEAT-AI `NeuronStateInterface`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NeuronTraceState {
    /// Records that accumulated a bias proposal for this neuron.
    pub count: f64,
    /// Accumulated raw bias mass.
    pub total_bias: f64,
    /// Accumulated adjusted-bias mass (the bias proposal's numerator).
    pub total_adjusted_bias: f64,
    /// Hint value of the last record.
    pub hint_value: f64,
    /// Largest activation seen over the pass.
    pub maximum_activation: f64,
    /// Smallest activation seen over the pass.
    pub minimum_activation: f64,
    /// Sum of the activations seen over the pass.
    pub total_activation: f64,
    /// Accumulated absolute error attributed to this neuron.
    pub total_error_absolute: f64,
    /// Whether the neuron was flagged `noChange` (outside the sparse selection).
    pub no_change: bool,
}

/// Per-synapse trace state — NEAT-AI `SynapseState`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SynapseTraceState {
    /// Records that accumulated a weight proposal for this synapse.
    pub count: f64,
    /// Positive activation mass.
    pub total_positive_activation: f64,
    /// Negative activation mass.
    pub total_negative_activation: f64,
    /// Records whose activation was positive.
    pub count_positive_activations: f64,
    /// Records whose activation was negative.
    pub count_negative_activations: f64,
    /// Positive adjusted-value mass.
    pub total_positive_adjusted_value: f64,
    /// Negative adjusted-value mass.
    pub total_negative_adjusted_value: f64,
}

/// Neuron types NEAT-AI's `traceJSON()` never attaches state to.
const UNTRACED_NEURON_TYPE: &str = "constant";

/// Build the NEAT-AI `CreatureTrace` for `creature` from an accumulate pass.
///
/// `creature` supplies the genes and `report` the accumulated state, so a
/// rejected epoch can pair the candidate that was measured worse with the
/// accumulation that proposed it.
///
/// Fails when the accumulate pass does not line up with the creature — a
/// mismatched gene count means the trace would silently mislabel state, so it
/// is an error rather than a partially populated artifact.
pub fn build_creature_trace(
    creature: &CreatureExport,
    report: &AccumulateReport,
) -> Result<Value, String> {
    // A trace is a creature export on disk — never write one without an
    // observation width (issue #92).
    crate::creature_io::check_observation_width(creature.input, creature.output)?;
    if report.neuron_traces.len() != creature.neurons.len() {
        return Err(format!(
            "trace has {} neuron rows for a creature of {} neurons",
            report.neuron_traces.len(),
            creature.neurons.len()
        ));
    }
    let mut value = serde_json::to_value(creature).map_err(|e| e.to_string())?;
    attach_neuron_traces(
        &mut value,
        creature,
        &report.learning,
        &report.neuron_traces,
    )?;
    attach_synapse_traces(&mut value, &report.learning)?;
    Ok(value)
}

/// Add `trace` to every neuron the pass accumulated.
fn attach_neuron_traces(
    value: &mut Value,
    creature: &CreatureExport,
    learning: &LearningSignal,
    stats: &[NeuronTraceStats],
) -> Result<(), String> {
    let rows = value
        .get_mut("neurons")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "serialised creature has no neurons array".to_string())?;
    for (i, row) in rows.iter_mut().enumerate() {
        let bias = match learning.biases.get(i) {
            Some(bias) if bias.count > 0.0 => bias,
            _ => continue,
        };
        let is_constant = creature
            .neurons
            .get(i)
            .is_some_and(|n| n.neuron_type == UNTRACED_NEURON_TYPE);
        if is_constant {
            continue;
        }
        let stat = stats.get(i).cloned().unwrap_or_default();
        let state = NeuronTraceState {
            count: bias.count,
            total_bias: stat.total_bias,
            total_adjusted_bias: bias.total_adjusted_bias,
            hint_value: stat.hint_value,
            maximum_activation: stat.maximum_activation,
            minimum_activation: stat.minimum_activation,
            total_activation: stat.total_activation,
            total_error_absolute: stat.total_error_absolute,
            no_change: bias.no_change,
        };
        insert_trace(row, &state)?;
    }
    Ok(())
}

/// Add `trace` to every synapse the pass accumulated.
fn attach_synapse_traces(value: &mut Value, learning: &LearningSignal) -> Result<(), String> {
    let rows = value
        .get_mut("synapses")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "serialised creature has no synapses array".to_string())?;
    for (i, row) in rows.iter_mut().enumerate() {
        let weight = match learning.weights.get(i) {
            Some(weight) if weight.count > 0.0 => weight,
            _ => continue,
        };
        let state = SynapseTraceState {
            count: weight.count,
            total_positive_activation: weight.total_positive_activation,
            total_negative_activation: weight.total_negative_activation,
            count_positive_activations: weight.count_positive,
            count_negative_activations: weight.count_negative,
            total_positive_adjusted_value: weight.total_positive_adjusted_value,
            total_negative_adjusted_value: weight.total_negative_adjusted_value,
        };
        insert_trace(row, &state)?;
    }
    Ok(())
}

/// Set `row.trace` to the serialised state.
fn insert_trace(row: &mut Value, state: &impl Serialize) -> Result<(), String> {
    let object = row
        .as_object_mut()
        .ok_or_else(|| "serialised gene is not a JSON object".to_string())?;
    object.insert(
        "trace".to_string(),
        serde_json::to_value(state).map_err(|e| e.to_string())?,
    );
    Ok(())
}

/// Write a trace artifact, creating the parent directory.
pub fn write_creature_trace(path: &Path, trace: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(trace).map_err(|e| e.to_string())?;
    fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backprop::{BiasSignal, WeightSignal};
    use neat_core::parse_creature_json;

    const CHAIN: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
      "neurons":[
        {"type":"constant","uuid":"c1","bias":0.5,"squash":"IDENTITY"},
        {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0}
      ]
    }"#;

    /// A report whose first synapse and the `c1` / `h1` neurons accumulated.
    fn report(creature: &CreatureExport) -> AccumulateReport {
        let mut learning = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
        learning.biases[0] = BiasSignal {
            count: 3.0,
            total_adjusted_bias: 0.75,
            no_change: false,
        };
        learning.biases[1] = BiasSignal {
            count: 3.0,
            total_adjusted_bias: 1.5,
            no_change: false,
        };
        learning.weights[0] = WeightSignal {
            count: 3.0,
            total_positive_activation: 6.0,
            count_positive: 3.0,
            total_positive_adjusted_value: 9.0,
            ..WeightSignal::default()
        };
        let mut neuron_traces = vec![NeuronTraceStats::default(); creature.neurons.len()];
        neuron_traces[1] = NeuronTraceStats {
            records: 3,
            total_bias: 0.6,
            total_error_absolute: 0.9,
            total_activation: 4.5,
            maximum_activation: 2.0,
            minimum_activation: 1.0,
            hint_value: 1.5,
        };
        AccumulateReport {
            learning,
            neuron_traces,
            mse: 0.25,
            records: 3,
        }
    }

    #[test]
    fn trace_carries_uuid_endpoints_and_accumulated_state() {
        let creature = parse_creature_json(CHAIN).unwrap();
        let trace = build_creature_trace(&creature, &report(&creature)).unwrap();

        let neurons = trace["neurons"].as_array().unwrap();
        assert_eq!(neurons.len(), 3, "every export neuron survives");
        assert_eq!(neurons[1]["uuid"], "h1");
        let state: NeuronTraceState = serde_json::from_value(neurons[1]["trace"].clone()).unwrap();
        assert_eq!(state.count, 3.0);
        assert_eq!(state.total_adjusted_bias, 1.5);
        assert_eq!(state.total_bias, 0.6);
        assert_eq!(state.maximum_activation, 2.0);
        assert_eq!(state.minimum_activation, 1.0);
        assert_eq!(state.total_activation, 4.5);
        assert_eq!(state.hint_value, 1.5);
        assert_eq!(state.total_error_absolute, 0.9);
        assert!(!state.no_change);

        let synapses = trace["synapses"].as_array().unwrap();
        assert_eq!(synapses[0]["fromUUID"], "input-0");
        assert_eq!(synapses[0]["toUUID"], "h1");
        let state: SynapseTraceState =
            serde_json::from_value(synapses[0]["trace"].clone()).unwrap();
        assert_eq!(state.count, 3.0);
        assert_eq!(state.total_positive_activation, 6.0);
        assert_eq!(state.count_positive_activations, 3.0);
        assert_eq!(state.total_positive_adjusted_value, 9.0);
    }

    #[test]
    fn genes_without_accumulation_carry_no_trace() {
        let creature = parse_creature_json(CHAIN).unwrap();
        let trace = build_creature_trace(&creature, &report(&creature)).unwrap();
        // `o1` never accumulated, and NEAT-AI never traces a constant neuron
        // even when it did.
        assert!(trace["neurons"][2].get("trace").is_none());
        assert!(trace["neurons"][0].get("trace").is_none());
        assert!(trace["synapses"][1].get("trace").is_none());
    }

    /// The export half must stay byte-identical to the creature, so a trace
    /// loads as a creature as well as a debugging artifact.
    #[test]
    fn trace_preserves_the_creature_export() {
        let creature = parse_creature_json(CHAIN).unwrap();
        let mut trace = build_creature_trace(&creature, &report(&creature)).unwrap();
        for key in ["neurons", "synapses"] {
            for row in trace[key].as_array_mut().unwrap() {
                row.as_object_mut().unwrap().remove("trace");
            }
        }
        assert_eq!(trace, serde_json::to_value(&creature).unwrap());
    }

    /// A mismatched pass is a programming error — fail loud rather than write
    /// a trace whose state belongs to a different creature.
    #[test]
    fn mismatched_report_is_rejected() {
        let creature = parse_creature_json(CHAIN).unwrap();
        let mut report = report(&creature);
        report.neuron_traces.pop();
        let err = build_creature_trace(&creature, &report).unwrap_err();
        assert!(err.contains("neuron rows"), "{err}");
    }

    #[test]
    fn write_creates_the_store_directory() {
        let dir = tempfile::tempdir().unwrap();
        let creature = parse_creature_json(CHAIN).unwrap();
        let trace = build_creature_trace(&creature, &report(&creature)).unwrap();
        let path = dir.path().join("store").join("failed").join("epoch-1.json");
        write_creature_trace(&path, &trace).unwrap();
        let round_tripped: Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(round_tripped, trace);
    }
}
