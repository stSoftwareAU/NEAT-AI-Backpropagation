//! Rust port surface for NEAT-AI backpropagation configuration and learning
//! application helpers.
//!
//! The heavy reverse-topological loop lives in `neat-core`
//! ([`neat_core::propagate_topological_loop`]). This module ports the
//! TypeScript `BackPropagation` configuration / learning-rate semantics and
//! exposes analyse-without-apply accumulation helpers that finalise bias and
//! weight proposals via [`neat_core::calculate_bias`] /
//! [`neat_core::calculate_weight`].
//!
//! Lamarck defaults use fixed `generations: 1.0` (not the TS random 1–10 draw)
//! so optimisation runs stay deterministic under a seeded RNG.

use neat_core::{
    CreatureExport, PropagateOutcome, PropagateOutput, StandardOutcome, SynapseDelta,
    calculate_bias, calculate_weight,
};
use serde::{Deserialize, Serialize};

/// Floating-point absolute tolerance used by parity-style unit tests.
pub const FLOAT_ABS_TOL: f64 = 1e-9;

/// Floating-point relative tolerance used by parity-style unit tests.
pub const FLOAT_REL_TOL: f64 = 1e-6;

/// Learning-rate schedule strategies matching the TypeScript port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LearningRateStrategy {
    /// Constant learning rate.
    #[default]
    Fixed,
    /// Multiplicative decay each iteration.
    Decay,
    /// Boost when error improves / stagnates.
    Adaptive,
    /// Decay with periodic warm restart.
    WarmRestart,
}

/// Backpropagation configuration — behavioural port of NEAT-AI defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackpropConfig {
    /// Generational dampening weight.
    pub generations: f64,
    /// Active learning rate (may be overridden by strategy).
    pub learning_rate: f64,
    /// Maximum bias adjustment magnitude per apply.
    pub maximum_bias_adjustment_scale: f64,
    /// Maximum weight adjustment magnitude per apply.
    pub maximum_weight_adjustment_scale: f64,
    /// Hard bias magnitude limit.
    pub limit_bias_scale: f64,
    /// Hard weight magnitude limit.
    pub limit_weight_scale: f64,
    /// Minimum meaningful magnitude (plank constant).
    pub plank_constant: f64,
    /// Disable bias updates.
    pub disable_bias_adjustment: bool,
    /// Disable weight updates.
    pub disable_weight_adjustment: bool,
    /// Learning-rate strategy.
    pub learning_rate_strategy: LearningRateStrategy,
    /// Initial learning rate for decay/adaptive/warm_restart.
    pub initial_learning_rate: f64,
    /// Decay factor for decay / warm_restart.
    pub learning_rate_decay: f64,
    /// L1 bias decay.
    pub l1_bias_decay: f64,
    /// L2 bias decay.
    pub l2_bias_decay: f64,
    /// L1 weight decay.
    pub l1_weight_decay: f64,
    /// L2 weight decay.
    pub l2_weight_decay: f64,
    /// Fraction of eligible (hidden/output) neurons selected for sparse updates.
    /// `1.0` = full network (Lamarck default for strongest focus signals).
    pub sparse_ratio: f64,
    /// When true, skip TS-style random generation sampling (already fixed here).
    pub disable_random_samples: bool,
    /// Divide multi-path gradients by `sqrt(path_count)` (NEAT-AI #1872).
    pub normalise_gradients: bool,
    /// Probability of mutating a gene during sparse clustering (TS `trainingMutationRate`).
    pub training_mutation_rate: f64,
}

impl Default for BackpropConfig {
    fn default() -> Self {
        Self {
            // Fixed (not TS random 1–10) for deterministic Lamarck runs.
            generations: 1.0,
            learning_rate: 0.01,
            maximum_bias_adjustment_scale: 10.0,
            maximum_weight_adjustment_scale: 10.0,
            limit_bias_scale: 10_000.0,
            limit_weight_scale: 100_000.0,
            plank_constant: 1e-7,
            disable_bias_adjustment: false,
            disable_weight_adjustment: false,
            learning_rate_strategy: LearningRateStrategy::Fixed,
            initial_learning_rate: 0.01,
            learning_rate_decay: 0.95,
            l1_bias_decay: 0.0,
            l2_bias_decay: 0.0,
            l1_weight_decay: 0.0,
            l2_weight_decay: 0.0,
            sparse_ratio: 1.0,
            disable_random_samples: true,
            normalise_gradients: false,
            training_mutation_rate: 1.0,
        }
    }
}

/// Compute the learning rate for an iteration using the configured strategy.
pub fn calculate_learning_rate(
    config: &BackpropConfig,
    iteration: u64,
    error_feedback: Option<(f64, f64)>,
) -> f64 {
    match config.learning_rate_strategy {
        LearningRateStrategy::Fixed => config.learning_rate,
        LearningRateStrategy::Decay => {
            config.initial_learning_rate * config.learning_rate_decay.powi(iteration as i32)
        }
        LearningRateStrategy::WarmRestart => {
            let cycle = 10u64;
            let pos = iteration % cycle;
            config.initial_learning_rate * config.learning_rate_decay.powi(pos as i32)
        }
        LearningRateStrategy::Adaptive => {
            let mut lr = config.initial_learning_rate;
            if let Some((prev, curr)) = error_feedback
                && prev > 0.0
            {
                let ratio = curr / prev;
                if ratio < 0.95 {
                    lr *= 1.1;
                } else if ratio >= 1.0 {
                    lr *= 1.3;
                } else {
                    lr *= ratio.max(0.5);
                }
            }
            lr.clamp(1e-8, 1.0)
        }
    }
}

/// Accumulated bias learning signal for one neuron (analyse-without-apply).
#[derive(Debug, Clone, Default)]
pub struct BiasSignal {
    /// Accumulation count.
    pub count: f64,
    /// Sum of adjusted bias contributions.
    pub total_adjusted_bias: f64,
    /// Whether the neuron flagged no-change.
    pub no_change: bool,
}

impl BiasSignal {
    /// Fold a standard backprop outcome into this accumulator.
    pub fn accumulate_standard(&mut self, outcome: &StandardOutcome) {
        self.count += f64::from(outcome.bias_count_delta);
        self.total_adjusted_bias += f64::from(outcome.total_adjusted_bias_delta);
        self.no_change |= outcome.no_change;
    }

    /// Propose a new bias without mutating the creature.
    pub fn propose(&self, current_bias: f64, config: &BackpropConfig, learning_rate: f64) -> f64 {
        if config.disable_bias_adjustment {
            return current_bias;
        }
        calculate_bias(
            self.count,
            self.total_adjusted_bias,
            current_bias,
            self.no_change,
            config.generations,
            config.plank_constant,
            learning_rate,
            config.maximum_bias_adjustment_scale,
            config.limit_bias_scale,
            config.l1_bias_decay,
            config.l2_bias_decay,
        )
    }
}

/// Accumulated weight learning signal for one synapse (analyse-without-apply).
#[derive(Debug, Clone, Default)]
pub struct WeightSignal {
    /// Accumulation count.
    pub count: f64,
    /// Positive activation mass.
    pub total_positive_activation: f64,
    /// Negative activation mass.
    pub total_negative_activation: f64,
    /// Positive activation count.
    pub count_positive: f64,
    /// Negative activation count.
    pub count_negative: f64,
    /// Positive adjusted-value mass.
    pub total_positive_adjusted_value: f64,
    /// Negative adjusted-value mass.
    pub total_negative_adjusted_value: f64,
}

impl WeightSignal {
    /// Fold a synapse delta into this accumulator.
    pub fn accumulate_delta(&mut self, delta: &SynapseDelta) {
        self.count += f64::from(delta.count);
        self.total_positive_activation += f64::from(delta.total_positive_activation);
        self.total_negative_activation += f64::from(delta.total_negative_activation);
        self.count_positive += f64::from(delta.count_positive);
        self.count_negative += f64::from(delta.count_negative);
        self.total_positive_adjusted_value += f64::from(delta.total_positive_adjusted_value);
        self.total_negative_adjusted_value += f64::from(delta.total_negative_adjusted_value);
    }

    /// Propose a new weight without mutating the creature.
    pub fn propose(&self, current_weight: f64, config: &BackpropConfig, learning_rate: f64) -> f64 {
        if config.disable_weight_adjustment {
            return current_weight;
        }
        calculate_weight(
            self.count,
            self.total_positive_activation,
            self.total_negative_activation,
            self.count_positive,
            self.count_negative,
            self.total_positive_adjusted_value,
            self.total_negative_adjusted_value,
            current_weight,
            config.generations,
            config.plank_constant,
            learning_rate,
            config.maximum_weight_adjustment_scale,
            config.limit_weight_scale,
            config.l1_weight_decay,
            config.l2_weight_decay,
        )
    }
}

/// Aggregated learning signal for a creature — never applied automatically.
#[derive(Debug, Clone, Default)]
pub struct LearningSignal {
    /// Per-neuron bias signals indexed by `CreatureExport.neurons` position
    /// (non-input neurons only — matches candidate generation).
    pub biases: Vec<BiasSignal>,
    /// Per-synapse weight signals indexed by `CreatureExport.synapses` position.
    pub weights: Vec<WeightSignal>,
}

impl LearningSignal {
    /// Create empty accumulators sized for export neurons/synapses.
    pub fn new(neuron_count: usize, synapse_count: usize) -> Self {
        Self {
            biases: vec![BiasSignal::default(); neuron_count],
            weights: vec![WeightSignal::default(); synapse_count],
        }
    }

    /// Fold one full-layout [`PropagateOutput`] into export-indexed signals.
    ///
    /// Propagate neuron indices `0..input_count` are virtual inputs and skipped;
    /// `input_count + i` maps to `biases[i]`. Synapses share export order.
    pub fn accumulate_propagate_output(&mut self, output: &PropagateOutput, input_count: usize) {
        for (prop_idx, outcome) in output.neurons.iter().enumerate() {
            if prop_idx < input_count {
                continue;
            }
            let export_idx = prop_idx - input_count;
            if let Some(signal) = self.biases.get_mut(export_idx)
                && let PropagateOutcome::Standard(standard) = outcome
            {
                signal.accumulate_standard(standard);
            }
        }
        for (idx, delta) in output.synapses.iter().enumerate() {
            if let Some(signal) = self.weights.get_mut(idx) {
                signal.accumulate_delta(delta);
            }
        }
    }
}

/// How [`apply_learnings_with`] writes proposals onto the clone.
#[derive(Debug, Clone, Copy)]
pub struct ApplyOptions {
    /// Multiply `(proposed − current)` by this factor (`1.0` = full step).
    pub step_scale: f64,
    /// When true, only output neurons and synapses that target an output.
    pub outputs_only: bool,
    /// When true, skip output neurons and synapses that target an output.
    pub hidden_only: bool,
}

impl Default for ApplyOptions {
    fn default() -> Self {
        Self {
            step_scale: 1.0,
            outputs_only: false,
            hidden_only: false,
        }
    }
}

/// Apply accumulated learning proposals to a cloned creature (analyse ≠ apply).
///
/// Skips neurons/synapses with zero accumulation count and changes smaller than
/// `plank_constant`. Does not mutate `creature`.
pub fn apply_learnings(
    creature: &CreatureExport,
    signal: &LearningSignal,
    config: &BackpropConfig,
    learning_rate: f64,
) -> CreatureExport {
    apply_learnings_with(
        creature,
        signal,
        config,
        learning_rate,
        ApplyOptions::default(),
    )
}

/// [`apply_learnings`] with an explicit step scale / output-only filter.
pub fn apply_learnings_with(
    creature: &CreatureExport,
    signal: &LearningSignal,
    config: &BackpropConfig,
    learning_rate: f64,
    options: ApplyOptions,
) -> CreatureExport {
    let step = if options.step_scale.is_finite() && options.step_scale > 0.0 {
        options.step_scale.min(1.0)
    } else {
        1.0
    };
    let output_uuids: std::collections::HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();
    let mut out = creature.clone();
    for (i, neuron) in out.neurons.iter_mut().enumerate() {
        if options.outputs_only && neuron.neuron_type != "output" {
            continue;
        }
        if options.hidden_only && neuron.neuron_type == "output" {
            continue;
        }
        let Some(bias_sig) = signal.biases.get(i) else {
            continue;
        };
        if bias_sig.count <= 0.0 {
            continue;
        }
        let proposed = bias_sig.propose(neuron.bias, config, learning_rate);
        let next = neuron.bias + (proposed - neuron.bias) * step;
        if (next - neuron.bias).abs() >= config.plank_constant {
            neuron.bias = next;
        }
    }
    for (i, syn) in out.synapses.iter_mut().enumerate() {
        let targets_output = output_uuids.contains(syn.to_uuid.as_str());
        if options.outputs_only && !targets_output {
            continue;
        }
        if options.hidden_only && targets_output {
            continue;
        }
        let Some(w_sig) = signal.weights.get(i) else {
            continue;
        };
        if w_sig.count <= 0.0 {
            continue;
        }
        let proposed = w_sig.propose(syn.weight, config, learning_rate);
        let next = syn.weight + (proposed - syn.weight) * step;
        if (next - syn.weight).abs() >= config.plank_constant {
            syn.weight = next;
        }
    }
    out
}

/// Count how many export genes actually moved in an apply.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApplyDeltaCounts {
    /// Hidden / constant neurons whose bias changed.
    pub hidden_biases: usize,
    /// Output neurons whose bias changed.
    pub output_biases: usize,
    /// Synapses that do not target an output.
    pub hidden_weights: usize,
    /// Synapses that target an output.
    pub output_weights: usize,
}

/// Diff two creature exports after an apply (plank-thresholded).
pub fn count_apply_deltas(
    before: &CreatureExport,
    after: &CreatureExport,
    plank: f64,
) -> ApplyDeltaCounts {
    let output_uuids: std::collections::HashSet<&str> = before
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();
    let mut counts = ApplyDeltaCounts::default();
    for (a, b) in before.neurons.iter().zip(after.neurons.iter()) {
        if (a.bias - b.bias).abs() < plank {
            continue;
        }
        if a.neuron_type == "output" {
            counts.output_biases += 1;
        } else {
            counts.hidden_biases += 1;
        }
    }
    for (a, b) in before.synapses.iter().zip(after.synapses.iter()) {
        if (a.weight - b.weight).abs() < plank {
            continue;
        }
        if output_uuids.contains(a.to_uuid.as_str()) {
            counts.output_weights += 1;
        } else {
            counts.hidden_weights += 1;
        }
    }
    counts
}

/// Compare two floats with ordinary absolute/relative tolerances.
pub fn nearly_equal(a: f64, b: f64) -> bool {
    let diff = (a - b).abs();
    if diff <= FLOAT_ABS_TOL {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    diff <= FLOAT_REL_TOL * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_learning_rate_is_constant() {
        let cfg = BackpropConfig::default();
        assert!(nearly_equal(calculate_learning_rate(&cfg, 0, None), 0.01));
        assert!(nearly_equal(calculate_learning_rate(&cfg, 100, None), 0.01));
    }

    #[test]
    fn decay_learning_rate_shrinks() {
        let cfg = BackpropConfig {
            learning_rate_strategy: LearningRateStrategy::Decay,
            initial_learning_rate: 0.1,
            learning_rate_decay: 0.5,
            ..Default::default()
        };
        let lr0 = calculate_learning_rate(&cfg, 0, None);
        let lr2 = calculate_learning_rate(&cfg, 2, None);
        assert!(nearly_equal(lr0, 0.1));
        assert!(nearly_equal(lr2, 0.025));
    }

    #[test]
    fn bias_propose_steps_towards_accumulated_mean_without_overshooting() {
        let cfg = BackpropConfig::default();
        let signal = BiasSignal {
            count: 10.0,
            total_adjusted_bias: 5.0,
            no_change: false,
        };
        // Accumulated evidence says the bias should sit near 5/10; a single
        // proposal is a partial step towards it, never past it.
        let mean = signal.total_adjusted_bias / signal.count;
        let proposed = signal.propose(0.0, &cfg, 0.01);
        assert!(
            proposed > 0.0 && proposed < mean,
            "expected 0 < {proposed} < {mean}"
        );

        let negative = BiasSignal {
            total_adjusted_bias: -5.0,
            ..signal.clone()
        };
        let proposed_negative = negative.propose(0.0, &cfg, 0.01);
        assert!(
            proposed_negative < 0.0 && proposed_negative > -mean,
            "expected {} < {proposed_negative} < 0",
            -mean
        );

        // A larger learning rate takes a larger step towards the same mean.
        let faster = signal.propose(0.0, &cfg, 0.02);
        assert!(faster > proposed && faster < mean, "expected step to grow");
    }

    #[test]
    fn bias_propose_step_never_exceeds_maximum_bias_adjustment_scale() {
        let cfg = BackpropConfig {
            maximum_bias_adjustment_scale: 0.25,
            ..Default::default()
        };
        let signal = BiasSignal {
            count: 10.0,
            total_adjusted_bias: 1_000_000.0,
            no_change: false,
        };
        let current = 0.0;
        let proposed = signal.propose(current, &cfg, 0.01);
        assert!(
            proposed > current,
            "an enormous signal still moves the bias"
        );
        assert!(
            (proposed - current).abs() <= cfg.maximum_bias_adjustment_scale + FLOAT_ABS_TOL,
            "step {} exceeded the clamp {}",
            proposed - current,
            cfg.maximum_bias_adjustment_scale
        );
    }

    #[test]
    fn bias_propose_returns_current_bias_when_adjustment_disabled() {
        let cfg = BackpropConfig {
            disable_bias_adjustment: true,
            ..Default::default()
        };
        let signal = BiasSignal {
            count: 10.0,
            total_adjusted_bias: 5.0,
            no_change: false,
        };
        assert_eq!(signal.propose(0.75, &cfg, 0.01), 0.75);
    }

    #[test]
    fn bias_propose_returns_current_bias_without_usable_signal() {
        let cfg = BackpropConfig::default();
        let empty = BiasSignal::default();
        assert_eq!(empty.propose(0.75, &cfg, 0.01), 0.75);

        let flagged = BiasSignal {
            count: 10.0,
            total_adjusted_bias: 5.0,
            no_change: true,
        };
        assert_eq!(flagged.propose(0.75, &cfg, 0.01), 0.75);
    }

    /// Signal whose positive activation mass carries `adjusted` units of
    /// adjusted value per unit of activation — i.e. the accumulated evidence
    /// argues the weight should be `adjusted`.
    fn positive_weight_signal(adjusted: f64) -> WeightSignal {
        WeightSignal {
            count: 4.0,
            total_positive_activation: 2.0,
            total_negative_activation: 0.0,
            count_positive: 4.0,
            count_negative: 0.0,
            total_positive_adjusted_value: adjusted * 2.0,
            total_negative_adjusted_value: 0.0,
        }
    }

    #[test]
    fn weight_propose_steps_towards_accumulated_target_without_overshooting() {
        let cfg = BackpropConfig::default();
        let current = 0.5;

        let upwards = positive_weight_signal(1.0).propose(current, &cfg, 0.01);
        assert!(
            upwards > current && upwards < 1.0,
            "expected {current} < {upwards} < 1.0"
        );

        let downwards = positive_weight_signal(0.0).propose(current, &cfg, 0.01);
        assert!(
            downwards < current && downwards > 0.0,
            "expected 0.0 < {downwards} < {current}"
        );

        let faster = positive_weight_signal(1.0).propose(current, &cfg, 0.02);
        assert!(faster > upwards && faster < 1.0, "expected step to grow");
    }

    #[test]
    fn weight_propose_step_never_exceeds_maximum_weight_adjustment_scale() {
        let cfg = BackpropConfig {
            maximum_weight_adjustment_scale: 0.25,
            ..Default::default()
        };
        let current = 0.5;
        let proposed = positive_weight_signal(500_000.0).propose(current, &cfg, 0.01);
        assert!(
            proposed > current,
            "an enormous signal still moves the weight"
        );
        assert!(
            (proposed - current).abs() <= cfg.maximum_weight_adjustment_scale + FLOAT_ABS_TOL,
            "step {} exceeded the clamp {}",
            proposed - current,
            cfg.maximum_weight_adjustment_scale
        );
    }

    #[test]
    fn weight_propose_returns_current_weight_when_adjustment_disabled() {
        let cfg = BackpropConfig {
            disable_weight_adjustment: true,
            ..Default::default()
        };
        assert_eq!(positive_weight_signal(1.0).propose(0.5, &cfg, 0.01), 0.5);
    }

    #[test]
    fn weight_propose_returns_current_weight_without_usable_signal() {
        let cfg = BackpropConfig::default();
        let empty = WeightSignal::default();
        assert_eq!(empty.propose(0.5, &cfg, 0.01), 0.5);

        // Counted accumulations carrying no activation mass are not evidence.
        let no_activation = WeightSignal {
            count: 4.0,
            count_positive: 4.0,
            ..WeightSignal::default()
        };
        assert_eq!(no_activation.propose(0.5, &cfg, 0.01), 0.5);
    }

    #[test]
    fn learning_signal_does_not_mutate_creature_on_accumulate() {
        let mut signal = LearningSignal::new(1, 1);
        let output = PropagateOutput {
            neurons: vec![
                PropagateOutcome::Skipped,
                PropagateOutcome::Standard(StandardOutcome {
                    bias_count_delta: 1,
                    total_adjusted_bias_delta: 0.25,
                    ..StandardOutcome::default()
                }),
            ],
            synapses: vec![SynapseDelta {
                count: 1.0,
                total_positive_activation: 1.0,
                count_positive: 1.0,
                total_positive_adjusted_value: 0.5,
                ..SynapseDelta::default()
            }],
        };
        // Full layout: [input0, export0] with input_count=1.
        signal.accumulate_propagate_output(&output, 1);
        assert!(nearly_equal(signal.biases[0].count, 1.0));
        assert!(nearly_equal(signal.weights[0].count, 1.0));
    }

    #[test]
    fn count_apply_deltas_sees_hidden_and_output() {
        use neat_core::parse_creature_json;
        let before = parse_creature_json(
            r#"{
              "input":1,"output":1,"forwardOnly":true,
              "neurons":[
                {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
                {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
              ],
              "synapses":[
                {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
                {"fromUUID":"h1","toUUID":"o1","weight":1.0}
              ]
            }"#,
        )
        .unwrap();
        let mut after = before.clone();
        after.neurons[0].bias = 0.5;
        after.neurons[1].bias = 0.25;
        after.synapses[0].weight = 1.1;
        after.synapses[1].weight = 0.9;
        let counts = count_apply_deltas(&before, &after, 1e-7);
        assert_eq!(counts.hidden_biases, 1);
        assert_eq!(counts.output_biases, 1);
        assert_eq!(counts.hidden_weights, 1);
        assert_eq!(counts.output_weights, 1);
    }

    #[test]
    fn apply_learnings_updates_bias_without_touching_source() {
        use neat_core::parse_creature_json;
        let creature = parse_creature_json(
            r#"{
              "input":1,"output":1,"forwardOnly":true,
              "neurons":[{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
              "synapses":[{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
            }"#,
        )
        .unwrap();
        let mut signal = LearningSignal::new(1, 1);
        signal.biases[0] = BiasSignal {
            count: 10.0,
            total_adjusted_bias: 5.0,
            no_change: false,
        };
        let cfg = BackpropConfig::default();
        let applied = apply_learnings(&creature, &signal, &cfg, 0.01);
        assert!((creature.neurons[0].bias - 0.0).abs() < 1e-15);
        assert!((applied.neurons[0].bias - creature.neurons[0].bias).abs() > cfg.plank_constant);
    }

    #[test]
    fn hidden_only_skips_output_genes() {
        use neat_core::parse_creature_json;
        let creature = parse_creature_json(
            r#"{
              "input":1,"output":1,"forwardOnly":true,
              "neurons":[
                {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
                {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
              ],
              "synapses":[
                {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
                {"fromUUID":"h1","toUUID":"o1","weight":1.0}
              ]
            }"#,
        )
        .unwrap();
        let mut signal = LearningSignal::new(2, 2);
        signal.biases[0] = BiasSignal {
            count: 10.0,
            total_adjusted_bias: 5.0,
            no_change: false,
        };
        signal.biases[1] = BiasSignal {
            count: 10.0,
            total_adjusted_bias: 5.0,
            no_change: false,
        };
        signal.weights[0] = WeightSignal {
            count: 4.0,
            total_positive_activation: 2.0,
            count_positive: 4.0,
            total_positive_adjusted_value: 1.0,
            ..WeightSignal::default()
        };
        signal.weights[1] = WeightSignal {
            count: 4.0,
            total_positive_activation: 2.0,
            count_positive: 4.0,
            total_positive_adjusted_value: 1.0,
            ..WeightSignal::default()
        };
        let applied = apply_learnings_with(
            &creature,
            &signal,
            &BackpropConfig::default(),
            0.01,
            ApplyOptions {
                step_scale: 1.0,
                outputs_only: false,
                hidden_only: true,
            },
        );
        let counts = count_apply_deltas(&creature, &applied, 1e-7);
        assert!(counts.hidden_biases + counts.hidden_weights > 0);
        assert_eq!(counts.output_biases, 0);
        assert_eq!(counts.output_weights, 0);
    }
}
