//! Evidence-driven sparse target selection (issue #108).
//!
//! `sparse_ratio` decides *how much* of the network participates in a pass;
//! this module decides *where* the expensive experiments are spent afterwards.
//! On an already highly evolved creature a scorer run is the scarce resource,
//! so drawing focus neurons uniformly at random spends most of them on genes
//! the accumulation pass gave no usable signal for.
//!
//! One accumulation pass already measures everything needed to rank targets:
//! the per-neuron trace ([`NeuronTraceStats`]) carries error mass, activation
//! count and activation range, and the per-gene accumulators carry the
//! proposal and the per-record direction split. [`rank_targets`] turns that
//! into a ranked candidate list, and [`select_targets`] draws from it under a
//! configurable exploitation vs random-control ratio.
//!
//! The ranking is a weighted sum of five normalised signals — the weights are
//! deliberately fixed and documented rather than tuned per corpus, because a
//! per-corpus weighting is exactly the private, domain-specific logic this
//! public repository must not carry:
//!
//! | Signal | Weight | Why |
//! | ------ | ------ | --- |
//! | Error mass | 0.35 | Accumulated absolute error is the evidence that something here is wrong. |
//! | Relative proposal | 0.30 | A proposal large *against the parameter it moves*, gated by the absolute move, is a real experiment rather than rounding against nothing. |
//! | Activity | 0.15 | A neuron few records produced learning for, or whose activation never moved, gives an unreliable gradient. |
//! | Consistency | 0.15 | Records pulling the same way are worth more than records cancelling out. |
//! | Degree | 0.05 | Fan-in / fan-out is a secondary structural tie-break, not evidence. |
//!
//! Longest-path depth is recorded as a rank feature but **not** scored: no
//! measurement in this crate establishes which direction depth should push, so
//! weighting it would be a guess dressed as evidence. Finite-difference sign
//! confidence is likewise absent: it needs the `gradient-check` probes, which
//! one accumulation pass does not produce.
//!
//! **The ranking is only as wide as the pass.** With `sparse_ratio < 1.0` the
//! pass accumulates for its own random subset alone, so error mass, proposal
//! and consistency are zero for every neuron outside it and the ranking
//! degenerates to "rank that subset" — [`crate::blockwise::run_blocks`] warns
//! when the two are combined.

use crate::backprop::LearningSignal;
use crate::blocks::{BlockGraph, ProposalMagnitudes};
use crate::gene_facets::CreatureTopology;
use crate::propagate_layout::NeuronTraceStats;
use neat_core::CreatureExport;
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};

/// Weight of the normalised accumulated error mass.
const WEIGHT_ERROR_MASS: f64 = 0.35;
/// Weight of the proposal magnitude relative to the parameter magnitude.
const WEIGHT_RELATIVE_PROPOSAL: f64 = 0.30;
/// Weight of the activation coverage / range term.
const WEIGHT_ACTIVITY: f64 = 0.15;
/// Weight of the per-record direction consistency.
const WEIGHT_CONSISTENCY: f64 = 0.15;
/// Weight of the secondary fan-in / fan-out term.
const WEIGHT_DEGREE: f64 = 0.05;

/// Seconds in an hour — the unit the throughput comparison reports in.
const SECONDS_PER_HOUR: f64 = 3600.0;

/// How target neurons are drawn from the ranked candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum TargetStrategy {
    /// Uniform random draw over every eligible neuron — the control arm the
    /// heuristic is measured against. (The behaviour before issue #108 was a
    /// deterministic proposal-magnitude ranking, which survives as one term of
    /// [`TargetStrategy::Evidence`]; this uniform draw is the baseline the
    /// issue asks the heuristic to beat, not that old behaviour.)
    Random,
    /// Highest-ranked evidence first.
    #[default]
    Evidence,
}

impl TargetStrategy {
    /// Stable lower-case slug for labels and CLI values.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Evidence => "evidence",
        }
    }

    /// The selector that implements this strategy.
    pub fn selector(self) -> &'static dyn TargetSelector {
        match self {
            Self::Random => &UniformRandomTargets,
            Self::Evidence => &EvidenceTargets,
        }
    }
}

/// Why one target was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TargetSource {
    /// Chosen because the accumulated evidence ranked it highest.
    Evidence,
    /// Drawn uniformly at random — the control arm that keeps the heuristic
    /// measurable and still finds accidental wins.
    RandomControl,
}

/// The evidence one accumulation pass carries about a candidate target.
///
/// Every field is recorded on the candidate, so a win (or a loss) can be read
/// back against the features that predicted it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TargetFeatures {
    /// Accumulated absolute error mass over the pass.
    pub error_mass: f64,
    /// |Δ| the accumulated learning proposes for the neuron's own genes.
    pub proposal_magnitude: f64,
    /// Current |parameter| of those same genes.
    pub parameter_magnitude: f64,
    /// `proposal / (proposal + parameter)` — a bounded "how big is this move
    /// against what it moves", `0.0` when neither is present.
    pub relative_proposal: f64,
    /// Records the accumulation pass folded into this neuron's trace.
    ///
    /// The pass observes every non-input neuron on every record, so this is
    /// the pass length rather than a per-neuron count — it is recorded for
    /// context and **not** scored. [`TargetFeatures::learning_records`] is the
    /// count that actually varies between targets.
    pub activation_records: u64,
    /// Records that produced a bias accumulation here — the neuron was inside
    /// the sparse selection, the pass reached it, and it did not flag
    /// `noChange`. This is the activation-count signal the ranking uses.
    pub learning_records: f64,
    /// Largest minus smallest activation seen.
    pub activation_range: f64,
    /// Share of per-record weight proposals pulling the same way, `0.0`–`1.0`.
    pub signal_consistency: f64,
    /// Inward synapse count.
    pub fan_in: usize,
    /// Outward synapse count.
    pub fan_out: usize,
    /// Longest-path depth from an input — recorded, not scored.
    pub depth: usize,
}

/// One candidate target, with its rank and the features it ranked on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RankedTarget {
    /// Position in [`CreatureExport::neurons`].
    pub neuron: usize,
    /// Zero-based rank over the eligible pool, best first.
    pub rank: usize,
    /// Weighted evidence score, `0.0`–`1.0`.
    pub score: f64,
    /// What the score was built from.
    pub features: TargetFeatures,
}

impl RankedTarget {
    /// Record this target as selected by `source`.
    pub fn selected_by(&self, source: TargetSource) -> SelectedTarget {
        SelectedTarget {
            neuron: self.neuron,
            selection: TargetSelection {
                source,
                rank: self.rank,
                score: self.score,
                features: self.features.clone(),
            },
        }
    }
}

/// Why a candidate's target was selected — the metadata written beside it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TargetSelection {
    /// Exploitation or random control.
    pub source: TargetSource,
    /// The target's rank in the evidence ranking.
    pub rank: usize,
    /// Its evidence score.
    pub score: f64,
    /// The features that produced the score.
    pub features: TargetFeatures,
}

/// A target neuron and the reason it was chosen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectedTarget {
    /// Position in [`CreatureExport::neurons`].
    pub neuron: usize,
    /// Why it was selected.
    pub selection: TargetSelection,
}

/// How targets are drawn — the strategy interface.
///
/// Implementations receive the ranking [`rank_targets`] produced and return the
/// targets to spend experiments on. `pool` is best-ranked first.
pub trait TargetSelector {
    /// Which arm the drawn targets belong to.
    fn source(&self) -> TargetSource;

    /// Draw at most `count` targets from `pool`, without repeating one.
    fn choose(
        &self,
        pool: &[RankedTarget],
        count: usize,
        rng: &mut dyn RngCore,
    ) -> Vec<SelectedTarget>;
}

/// Exploitation: the top of the evidence ranking, in order.
#[derive(Debug, Clone, Copy, Default)]
pub struct EvidenceTargets;

impl TargetSelector for EvidenceTargets {
    fn source(&self) -> TargetSource {
        TargetSource::Evidence
    }

    fn choose(
        &self,
        pool: &[RankedTarget],
        count: usize,
        _rng: &mut dyn RngCore,
    ) -> Vec<SelectedTarget> {
        if count == 0 {
            return Vec::new();
        }
        pool.iter()
            .take(count)
            .map(|t| t.selected_by(TargetSource::Evidence))
            .collect()
    }
}

/// The control arm: a uniform draw over the pool, ignoring the ranking.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformRandomTargets;

impl TargetSelector for UniformRandomTargets {
    fn source(&self) -> TargetSource {
        TargetSource::RandomControl
    }

    fn choose(
        &self,
        pool: &[RankedTarget],
        count: usize,
        rng: &mut dyn RngCore,
    ) -> Vec<SelectedTarget> {
        // A zero draw must not consume RNG: shuffling for no output would
        // shift the stream and make the two arms of one run irreproducible.
        if count == 0 {
            return Vec::new();
        }
        // Partial Fisher–Yates over an index list — the same shuffle
        // `select_sparse` uses, so a uniform draw here means what it means
        // there.
        let mut order: Vec<usize> = (0..pool.len()).collect();
        for i in (1..order.len()).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }
        order
            .into_iter()
            .take(count)
            .filter_map(|i| {
                pool.get(i)
                    .map(|t| t.selected_by(TargetSource::RandomControl))
            })
            .collect()
    }
}

/// Target-selection settings (issue #108).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TargetPlan {
    /// Which strategy draws the exploitation share.
    pub strategy: TargetStrategy,
    /// Share of each draw reserved for the uniform random control arm,
    /// `0.0`–`1.0`. Refused above `0.0` on [`TargetStrategy::Random`], whose
    /// whole draw is already the control — see [`TargetPlan::validate`].
    pub random_control_fraction: f64,
}

impl Default for TargetPlan {
    fn default() -> Self {
        Self {
            strategy: TargetStrategy::Evidence,
            random_control_fraction: 0.0,
        }
    }
}

impl TargetPlan {
    /// Refuse a plan that cannot mean what it says.
    ///
    /// A fraction outside `0.0..=1.0` (or a non-finite one) is a
    /// misconfiguration: silently clamping it would report a control arm that
    /// was never the size the operator asked for. A control fraction on a
    /// [`TargetStrategy::Random`] run is refused for the same reason `train`
    /// refuses scorer settings on an MSE run — the whole draw is already the
    /// control, so honouring the number is impossible and ignoring it would be
    /// silent.
    ///
    /// # Errors
    /// - `random_control_fraction` is non-finite or outside `0.0..=1.0`;
    /// - it is above `0.0` on the random strategy.
    pub fn validate(&self) -> Result<(), String> {
        let fraction = self.random_control_fraction;
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(format!(
                "randomControlFraction must be between 0.0 and 1.0 — got {fraction}"
            ));
        }
        if self.strategy == TargetStrategy::Random && fraction > 0.0 {
            return Err(format!(
                "randomControlFraction {fraction} is refused on the random strategy — its whole \
                 draw is already the control arm"
            ));
        }
        Ok(())
    }

    /// How many of `count` targets the control arm takes.
    ///
    /// A non-zero fraction always yields at least one control target, so a
    /// small draw cannot silently become pure exploitation — which also means
    /// a fraction below `1 / count` buys a larger control share than asked
    /// for; `--blocks-per-strategy` is what makes a small fraction realisable.
    ///
    /// `self` must have passed [`TargetPlan::validate`] — that is where an
    /// impossible fraction is refused loudly. The clamp here is defence in
    /// depth for a hand-built plan, never the primary check.
    pub fn control_share(&self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        if self.strategy == TargetStrategy::Random {
            return count;
        }
        let fraction = if self.random_control_fraction.is_finite() {
            self.random_control_fraction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if fraction <= 0.0 {
            return 0;
        }
        (((count as f64) * fraction).round() as usize).clamp(1, count)
    }
}

/// Rank `eligible` target neurons by the evidence one accumulation pass left.
///
/// `eligible` holds positions in [`CreatureExport::neurons`]; the returned list
/// is best first, with `rank` assigned over that pool. Ties keep export order,
/// so the ranking is deterministic. Every normalised term is scaled by the
/// pool's own maximum, so the score answers "loudest *here*" rather than
/// depending on the corpus size.
pub fn rank_targets(
    creature: &CreatureExport,
    graph: &BlockGraph,
    magnitudes: &ProposalMagnitudes,
    signal: &LearningSignal,
    traces: &[NeuronTraceStats],
    eligible: &[usize],
) -> Vec<RankedTarget> {
    let topology = CreatureTopology::of(creature);
    let mut features: Vec<(usize, TargetFeatures)> = Vec::with_capacity(eligible.len());
    for &index in eligible {
        features.push((
            index,
            features_of(
                creature, graph, magnitudes, signal, traces, &topology, index,
            ),
        ));
    }

    let max_error = pool_max(features.iter().map(|(_, f)| f.error_mass));
    let max_learning = pool_max(features.iter().map(|(_, f)| f.learning_records));
    let max_range = pool_max(features.iter().map(|(_, f)| f.activation_range));
    let max_proposal = pool_max(features.iter().map(|(_, f)| f.proposal_magnitude));
    let max_degree = pool_max(features.iter().map(|(_, f)| (f.fan_in + f.fan_out) as f64));

    let mut scored: Vec<(usize, f64, TargetFeatures)> = features
        .into_iter()
        .map(|(index, f)| {
            let coverage = normalise(f.learning_records, max_learning);
            let range = normalise(f.activation_range, max_range);
            // The ratio alone would crown a dead gene: an infinitesimal move
            // against a near-zero parameter is ratio 1.0. Gating it by the
            // pool-normalised absolute move keeps "big against what it moves"
            // and drops "big against nothing".
            let proposal = f.relative_proposal * normalise(f.proposal_magnitude, max_proposal);
            let score = WEIGHT_ERROR_MASS * normalise(f.error_mass, max_error)
                + WEIGHT_RELATIVE_PROPOSAL * proposal
                + WEIGHT_ACTIVITY * (0.5 * coverage + 0.5 * range)
                + WEIGHT_CONSISTENCY * f.signal_consistency
                + WEIGHT_DEGREE * normalise((f.fan_in + f.fan_out) as f64, max_degree);
            (index, score, f)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored
        .into_iter()
        .enumerate()
        .map(|(rank, (neuron, score, features))| RankedTarget {
            neuron,
            rank,
            score,
            features,
        })
        .collect()
}

/// Draw `count` targets under `plan`, splitting the draw between exploitation
/// and the uniform random control arm.
///
/// The control targets are drawn from what exploitation did **not** take, so a
/// control draw is never a relabelled exploitation pick — and, within one run,
/// never a *top-ranked* one either. That makes the in-run
/// [`SelectionComparison`] a comparison against the ranking's **tail**, which
/// flatters the evidence arm; the unbiased measurement is two runs, one per
/// [`TargetStrategy`], which is what
/// `scripts/run-target-selection-benchmark.sh` does.
///
/// # Errors
/// - `plan` is impossible — see [`TargetPlan::validate`].
pub fn select_targets(
    plan: &TargetPlan,
    ranked: &[RankedTarget],
    count: usize,
    rng: &mut dyn RngCore,
) -> Result<Vec<SelectedTarget>, String> {
    plan.validate()?;
    let count = count.min(ranked.len());
    let control = plan.control_share(count);
    let exploit = count - control;
    let mut selected = plan.strategy.selector().choose(ranked, exploit, rng);
    if control > 0 {
        let taken: Vec<usize> = selected.iter().map(|t| t.neuron).collect();
        let remaining: Vec<RankedTarget> = ranked
            .iter()
            .filter(|t| !taken.contains(&t.neuron))
            .cloned()
            .collect();
        selected.extend(UniformRandomTargets.choose(&remaining, control, rng));
    }
    Ok(selected)
}

/// One scored candidate's contribution to the arm comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmSample {
    /// Which arm selected the candidate's target.
    pub source: TargetSource,
    /// `candidate − incumbent` fitness, when the candidate was scored.
    pub score_delta: Option<f64>,
    /// Whether the gain cleared the win margin.
    pub win: bool,
    /// Wall-clock seconds the scorer run took.
    pub scorer_seconds: f64,
}

/// Scorer throughput of one selection arm (issue #108).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArmThroughput {
    /// Which arm this is.
    pub source: TargetSource,
    /// Candidates the scorer judged for this arm.
    pub candidates_scored: usize,
    /// Candidates whose gain cleared the win margin.
    pub wins: usize,
    /// Wall-clock scorer seconds this arm consumed.
    pub scorer_seconds: f64,
    /// Wins per hour of scorer time — `None` when the arm spent no time, so a
    /// rate is never divided out of nothing.
    pub wins_per_hour: Option<f64>,
    /// Sum of the **positive** score deltas: a rejected candidate is rolled
    /// back, so its loss is not a negative gain.
    pub total_score_gain: f64,
    /// Score gain per hour of scorer time.
    pub score_gain_per_hour: Option<f64>,
    /// Best single `candidate − incumbent` delta seen.
    pub best_score_delta: Option<f64>,
}

impl ArmThroughput {
    /// Fold `samples` belonging to `source` into one arm.
    fn of(source: TargetSource, samples: &[ArmSample]) -> Self {
        let mine = samples.iter().filter(|s| s.source == source);
        let mut candidates_scored = 0usize;
        let mut wins = 0usize;
        let mut scorer_seconds = 0.0f64;
        let mut total_score_gain = 0.0f64;
        let mut best_score_delta: Option<f64> = None;
        for sample in mine {
            candidates_scored += 1;
            wins += usize::from(sample.win);
            if sample.scorer_seconds.is_finite() && sample.scorer_seconds > 0.0 {
                scorer_seconds += sample.scorer_seconds;
            }
            if let Some(delta) = sample.score_delta.filter(|d| d.is_finite()) {
                if delta > 0.0 {
                    total_score_gain += delta;
                }
                best_score_delta =
                    Some(best_score_delta.map_or(delta, |best: f64| best.max(delta)));
            }
        }
        let per_hour =
            |value: f64| (scorer_seconds > 0.0).then(|| value * SECONDS_PER_HOUR / scorer_seconds);
        Self {
            source,
            candidates_scored,
            wins,
            scorer_seconds,
            wins_per_hour: per_hour(wins as f64),
            total_score_gain,
            score_gain_per_hour: per_hour(total_score_gain),
            best_score_delta,
        }
    }
}

/// Scorer throughput of the two arms, side by side (issue #108).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectionComparison {
    /// Targets the evidence ranking chose.
    pub evidence: ArmThroughput,
    /// Targets drawn uniformly at random.
    pub random_control: ArmThroughput,
}

/// Compare the scorer wins/hour and score gain/hour of the two arms.
///
/// Candidates whose block has no focus target — the whole-creature apply, the
/// output head, the top-genes block — carry no arm and are simply not
/// sampled: they answer "which region", not "which selection strategy".
pub fn compare_arms(samples: &[ArmSample]) -> SelectionComparison {
    SelectionComparison {
        evidence: ArmThroughput::of(TargetSource::Evidence, samples),
        random_control: ArmThroughput::of(TargetSource::RandomControl, samples),
    }
}

/// Measure one candidate target.
fn features_of(
    creature: &CreatureExport,
    graph: &BlockGraph,
    magnitudes: &ProposalMagnitudes,
    signal: &LearningSignal,
    traces: &[NeuronTraceStats],
    topology: &CreatureTopology,
    index: usize,
) -> TargetFeatures {
    let incident = graph.incident(index);
    let bias_delta = magnitudes.biases.get(index).copied().unwrap_or_default();
    let bias = creature.neurons.get(index).map_or(0.0, |n| n.bias);
    let mut proposal_magnitude = finite(bias_delta);
    let mut parameter_magnitude = finite(bias).abs();
    // Per-record direction split over the neuron's own synapses: records
    // pulling the same way are evidence, records cancelling out are noise.
    let mut directed = 0.0f64;
    let mut observed = 0.0f64;
    for &synapse in incident {
        proposal_magnitude += finite(magnitudes.weights.get(synapse).copied().unwrap_or_default());
        parameter_magnitude += creature
            .synapses
            .get(synapse)
            .map_or(0.0, |s| finite(s.weight).abs());
        if let Some(weight) = signal.weights.get(synapse) {
            let positive = finite(weight.count_positive);
            let negative = finite(weight.count_negative);
            directed += (positive - negative).abs();
            observed += positive + negative;
        }
    }
    let trace = traces.get(index);
    let records = trace.map_or(0, |t| t.records);
    let activation_range = trace
        .filter(|t| t.records > 0)
        .map_or(0.0, |t| finite(t.maximum_activation - t.minimum_activation))
        .max(0.0);
    let total = proposal_magnitude + parameter_magnitude;
    TargetFeatures {
        error_mass: trace.map_or(0.0, |t| finite(t.total_error_absolute).abs()),
        proposal_magnitude,
        parameter_magnitude,
        relative_proposal: if total > 0.0 {
            proposal_magnitude / total
        } else {
            0.0
        },
        activation_records: records,
        learning_records: signal.biases.get(index).map_or(0.0, |b| finite(b.count)),
        activation_range,
        signal_consistency: if observed > 0.0 {
            (directed / observed).clamp(0.0, 1.0)
        } else {
            0.0
        },
        fan_in: topology.fan_in(index),
        fan_out: topology.fan_out(index),
        depth: topology.depth(index),
    }
}

/// `value` when finite, `0.0` otherwise — a diverged accumulator must never
/// rank first on the strength of its own NaN.
fn finite(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

/// Largest finite value in the pool, or `0.0` for an empty pool.
fn pool_max(values: impl Iterator<Item = f64>) -> f64 {
    values.filter(|v| v.is_finite()).fold(0.0, f64::max)
}

/// `value / max`, clamped to `0.0..=1.0`; `0.0` when the pool has no maximum.
fn normalise(value: f64, max: f64) -> f64 {
    if max > 0.0 && value.is_finite() {
        (value / max).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backprop::{BackpropConfig, BiasSignal, WeightSignal, calculate_learning_rate};
    use crate::blocks::proposal_magnitudes;
    use neat_core::parse_creature_json;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Two hidden neurons feeding one output.
    const PAIR: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
      "neurons":[
        {"type":"hidden","uuid":"h1","bias":0.1,"squash":"IDENTITY"},
        {"type":"hidden","uuid":"h2","bias":0.1,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"input-0","toUUID":"h2","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":0.5},
        {"fromUUID":"h2","toUUID":"o1","weight":0.5}
      ]
    }"#;

    /// Ranked targets for `PAIR` with `h1` carrying every signal.
    fn ranked() -> (CreatureExport, Vec<RankedTarget>) {
        let creature = parse_creature_json(PAIR).unwrap();
        let config = BackpropConfig::default();
        let learning_rate = calculate_learning_rate(&config, 0, None);
        let mut signal = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
        signal.biases[0] = BiasSignal {
            count: 8.0,
            total_adjusted_bias: 24.0,
            no_change: false,
        };
        signal.weights[2] = WeightSignal {
            count: 8.0,
            total_positive_activation: 4.0,
            count_positive: 8.0,
            total_positive_adjusted_value: 12.0,
            ..WeightSignal::default()
        };
        let mut traces = vec![NeuronTraceStats::default(); creature.neurons.len()];
        traces[0] = NeuronTraceStats {
            records: 8,
            total_error_absolute: 4.0,
            maximum_activation: 1.0,
            minimum_activation: -1.0,
            ..NeuronTraceStats::default()
        };
        let graph = BlockGraph::of(&creature);
        let magnitudes = proposal_magnitudes(&creature, &signal, &config, learning_rate, 1.0);
        let ranked = rank_targets(&creature, &graph, &magnitudes, &signal, &traces, &[0, 1]);
        (creature, ranked)
    }

    #[test]
    fn the_scoring_weights_sum_to_one() {
        let sum = WEIGHT_ERROR_MASS
            + WEIGHT_RELATIVE_PROPOSAL
            + WEIGHT_ACTIVITY
            + WEIGHT_CONSISTENCY
            + WEIGHT_DEGREE;
        assert!((sum - 1.0).abs() < 1e-12, "weights sum to {sum}");
    }

    #[test]
    fn the_signalled_neuron_outranks_the_silent_one() {
        let (_, ranked) = ranked();
        assert_eq!(ranked[0].neuron, 0);
        assert!(ranked[0].score > ranked[1].score);
        assert!(ranked[0].score <= 1.0, "score is normalised");
        assert_eq!(ranked[1].features.error_mass, 0.0);
        assert_eq!(ranked[1].features.signal_consistency, 0.0);
    }

    #[test]
    fn a_non_finite_accumulator_cannot_rank_first() {
        let creature = parse_creature_json(PAIR).unwrap();
        let graph = BlockGraph::of(&creature);
        let mut magnitudes = ProposalMagnitudes {
            biases: vec![0.0; creature.neurons.len()],
            weights: vec![0.0; creature.synapses.len()],
        };
        magnitudes.biases[1] = f64::NAN;
        let signal = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
        let mut traces = vec![NeuronTraceStats::default(); creature.neurons.len()];
        traces[1].total_error_absolute = f64::INFINITY;
        traces[0] = NeuronTraceStats {
            records: 4,
            total_error_absolute: 1.0,
            maximum_activation: 1.0,
            minimum_activation: 0.0,
            ..NeuronTraceStats::default()
        };
        let ranked = rank_targets(&creature, &graph, &magnitudes, &signal, &traces, &[0, 1]);
        assert_eq!(ranked[0].neuron, 0, "the NaN neuron must not win the pool");
        assert!(ranked.iter().all(|t| t.score.is_finite()));
    }

    #[test]
    fn the_control_share_rounds_and_never_silently_vanishes() {
        let plan = TargetPlan {
            strategy: TargetStrategy::Evidence,
            random_control_fraction: 0.25,
        };
        assert_eq!(plan.control_share(4), 1);
        // Rounding to zero would turn an asked-for control arm into pure
        // exploitation, so a non-zero fraction always keeps one target.
        assert_eq!(plan.control_share(1), 1);
        assert_eq!(plan.control_share(0), 0);
        let none = TargetPlan {
            strategy: TargetStrategy::Evidence,
            random_control_fraction: 0.0,
        };
        assert_eq!(none.control_share(4), 0);
        let random = TargetPlan {
            strategy: TargetStrategy::Random,
            random_control_fraction: 0.0,
        };
        assert_eq!(random.control_share(4), 4, "a random run is all control");
    }

    #[test]
    fn a_draw_larger_than_the_pool_returns_the_pool_once() {
        let (_, ranked) = ranked();
        let plan = TargetPlan {
            strategy: TargetStrategy::Evidence,
            random_control_fraction: 0.5,
        };
        let mut rng = StdRng::seed_from_u64(2);
        let selected = select_targets(&plan, &ranked, 9, &mut rng).expect("valid plan");
        assert_eq!(selected.len(), 2);
        let mut neurons: Vec<usize> = selected.iter().map(|t| t.neuron).collect();
        neurons.sort_unstable();
        assert_eq!(neurons, vec![0, 1], "no target is drawn twice");
    }

    #[test]
    fn an_empty_pool_selects_nothing() {
        let plan = TargetPlan::default();
        let mut rng = StdRng::seed_from_u64(1);
        assert!(
            select_targets(&plan, &[], 4, &mut rng)
                .expect("valid plan")
                .is_empty()
        );
    }

    #[test]
    fn a_valid_plan_is_accepted_and_an_impossible_one_is_refused() {
        assert!(TargetPlan::default().validate().is_ok());
        assert!(
            TargetPlan {
                strategy: TargetStrategy::Random,
                random_control_fraction: 0.0,
            }
            .validate()
            .is_ok()
        );
        assert!(
            TargetPlan {
                strategy: TargetStrategy::Evidence,
                random_control_fraction: 2.0,
            }
            .validate()
            .is_err()
        );
    }

    /// A control fraction on a run that is already all control cannot be
    /// honoured, so it is refused rather than quietly dropped.
    #[test]
    fn a_control_fraction_on_the_random_strategy_is_refused() {
        let plan = TargetPlan {
            strategy: TargetStrategy::Random,
            random_control_fraction: 0.25,
        };
        let Err(err) = plan.validate() else {
            panic!("a control fraction on the random strategy must be refused");
        };
        assert!(err.contains("randomControlFraction"), "{err}");
        // The refusal reaches the caller through the draw, not only through a
        // validate() a caller might skip.
        let mut rng = StdRng::seed_from_u64(4);
        let (_, ranked) = ranked();
        assert!(select_targets(&plan, &ranked, 2, &mut rng).is_err());
    }

    /// The proposal term measures a real move, not a ratio: a gene whose
    /// parameters are ~zero has ratio 1.0 for an infinitesimal proposal, and
    /// must not outrank a gene proposing a genuinely large move.
    #[test]
    fn an_infinitesimal_move_against_nothing_cannot_outrank_a_real_one() {
        let creature = parse_creature_json(PAIR).unwrap();
        let graph = BlockGraph::of(&creature);
        let signal = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
        let traces = vec![NeuronTraceStats::default(); creature.neurons.len()];
        let mut magnitudes = ProposalMagnitudes {
            biases: vec![0.0; creature.neurons.len()],
            weights: vec![0.0; creature.synapses.len()],
        };
        // h1 proposes a real move against its own weights; h2 proposes almost
        // nothing, but its genes are zeroed so the ratio alone would crown it.
        magnitudes.biases[0] = 0.5;
        magnitudes.biases[1] = 1e-12;
        let mut creature = creature;
        creature.neurons[1].bias = 0.0;
        for synapse in &mut creature.synapses {
            if synapse.to_uuid == "h2" || synapse.from_uuid == "h2" {
                synapse.weight = 0.0;
            }
        }
        let ranked = rank_targets(&creature, &graph, &magnitudes, &signal, &traces, &[0, 1]);
        assert_eq!(ranked[0].neuron, 0, "the real move must rank first");
        assert!(
            ranked[1].features.relative_proposal > ranked[0].features.relative_proposal,
            "the ratio really is higher for the dead gene — the score is what must not follow it"
        );
    }

    /// The activation-count signal must discriminate. The trace's own record
    /// count does not (the pass folds every neuron on every record), so the
    /// ranking uses the bias accumulation count instead.
    #[test]
    fn the_activity_signal_uses_records_that_actually_produced_learning() {
        let creature = parse_creature_json(PAIR).unwrap();
        let graph = BlockGraph::of(&creature);
        let magnitudes = ProposalMagnitudes {
            biases: vec![0.0; creature.neurons.len()],
            weights: vec![0.0; creature.synapses.len()],
        };
        let mut signal = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
        signal.biases[1] = BiasSignal {
            count: 16.0,
            total_adjusted_bias: 1.0,
            no_change: false,
        };
        // Both neurons were folded into the trace on every record — only the
        // learning count separates them.
        let traces = vec![
            NeuronTraceStats {
                records: 16,
                ..NeuronTraceStats::default()
            };
            creature.neurons.len()
        ];
        let ranked = rank_targets(&creature, &graph, &magnitudes, &signal, &traces, &[0, 1]);
        assert_eq!(ranked[0].neuron, 1, "the neuron that learnt ranks first");
        assert_eq!(ranked[0].features.learning_records, 16.0);
        assert_eq!(ranked[1].features.learning_records, 0.0);
        assert_eq!(
            ranked[0].features.activation_records, ranked[1].features.activation_records,
            "the trace record count is the pass length, not a per-neuron signal"
        );
    }
}
