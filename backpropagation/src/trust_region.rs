//! Trust-region / update-budget limiting for a whole-creature apply (#109).
//!
//! `--step-scale` is a *per-gene* factor: every gene moves `step × (proposed −
//! current)`. That was a stable-sized optimisation step when the GRQ creature
//! carried ~16.6k parameters, but the same fixed scale on a creature of 7,363
//! neurons and 49,000 synapses moves a far larger aggregate distance — the
//! whole-creature perturbation grows with the number and magnitude of the
//! genes that move, not with the step scale alone.
//!
//! This module measures the update the applier is about to make
//! ([`measure_update`]) and, when a budget is configured, rescales that whole
//! update so its aggregate norms sit inside a [`TrustRegion`]. The step scale
//! stays an input: the trust region only ever *shrinks* it, never grows it, so
//! an unconfigured region reproduces the historical fixed-step apply exactly.

use crate::backprop::{
    ApplyOptions, BackpropConfig, LearningSignal, apply_learnings_with, effective_step_scale,
};
use neat_core::CreatureExport;
use serde::{Deserialize, Serialize};

/// Aggregate size of one proposed update over a set of genes.
///
/// Every field is measured over the genes that actually moved — a delta below
/// `plank_constant` is not written by the applier, so counting it would report
/// movement that never happened.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateNorms {
    /// Genes whose value moved by at least `plank_constant`.
    pub changed: usize,
    /// `Σ|Δ|` over the changed genes.
    pub l1: f64,
    /// `sqrt(Σ Δ²)` over the changed genes.
    pub l2: f64,
    /// `l2 / sqrt(changed)` — the typical per-gene move, `0` when nothing moved.
    pub rms: f64,
    /// Largest single `|Δ|`.
    pub max_abs: f64,
    /// `l2` divided by the L2 norm of the same genes' incumbent values, absent
    /// when those values are all zero and the ratio would be meaningless.
    #[serde(default)]
    pub relative_l2: Option<f64>,
    /// RMS of the per-gene relative change `Δ / value`, over the changed genes
    /// whose incumbent value clears `plank_constant`. Absent when no changed
    /// gene had a value to be relative to.
    #[serde(default)]
    pub relative_rms: Option<f64>,
}

/// Running totals behind one [`UpdateNorms`].
#[derive(Debug, Clone, Copy, Default)]
struct NormAccumulator {
    changed: usize,
    l1: f64,
    sum_sq: f64,
    max_abs: f64,
    base_sum_sq: f64,
    relative_sum_sq: f64,
    relative_count: usize,
}

impl NormAccumulator {
    /// Fold one moved gene in: its incumbent value and the delta applied to it.
    fn add(&mut self, value: f64, delta: f64, plank: f64) {
        self.changed += 1;
        self.l1 += delta.abs();
        self.sum_sq += delta * delta;
        self.max_abs = self.max_abs.max(delta.abs());
        self.base_sum_sq += value * value;
        if value.abs() > plank {
            let relative = delta / value;
            self.relative_sum_sq += relative * relative;
            self.relative_count += 1;
        }
    }

    /// Finish the accumulation into reportable norms.
    fn finish(self) -> UpdateNorms {
        let l2 = self.sum_sq.sqrt();
        UpdateNorms {
            changed: self.changed,
            l1: self.l1,
            l2,
            rms: if self.changed == 0 {
                0.0
            } else {
                (self.sum_sq / self.changed as f64).sqrt()
            },
            max_abs: self.max_abs,
            relative_l2: (self.base_sum_sq > 0.0).then(|| l2 / self.base_sum_sq.sqrt()),
            relative_rms: (self.relative_count > 0)
                .then(|| (self.relative_sum_sq / self.relative_count as f64).sqrt()),
        }
    }
}

/// One update, reported whole and split by gene class.
///
/// The classes are the two axes the trainer already reports movement counts
/// on: bias vs weight, and hidden vs output. `biases` + `weights` and
/// `hidden` + `output` each partition the same genes as `total`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateStats {
    /// Every gene that moved.
    pub total: UpdateNorms,
    /// Neuron biases only.
    pub biases: UpdateNorms,
    /// Synapse weights only.
    pub weights: UpdateNorms,
    /// Hidden / constant biases and synapses that do not target an output.
    pub hidden: UpdateNorms,
    /// Output biases and synapses that target an output.
    pub output: UpdateNorms,
}

/// Measure the update that turns `before` into `after`.
///
/// Fails loudly when the two creatures are not the same shape: a truncating
/// `zip` would silently under-report the update the applier is about to make,
/// which is exactly the number a budget is enforced against.
pub fn measure_update(
    before: &CreatureExport,
    after: &CreatureExport,
    plank: f64,
) -> Result<UpdateStats, String> {
    if before.neurons.len() != after.neurons.len() || before.synapses.len() != after.synapses.len()
    {
        return Err(format!(
            "cannot measure an update between creatures of different shape: \
             {} neurons / {} synapses before, {} / {} after",
            before.neurons.len(),
            after.neurons.len(),
            before.synapses.len(),
            after.synapses.len()
        ));
    }
    let output_uuids = output_uuids(before);
    let mut total = NormAccumulator::default();
    let mut biases = NormAccumulator::default();
    let mut weights = NormAccumulator::default();
    let mut hidden = NormAccumulator::default();
    let mut output = NormAccumulator::default();
    for (a, b) in before.neurons.iter().zip(after.neurons.iter()) {
        let delta = b.bias - a.bias;
        if delta.abs() < plank {
            continue;
        }
        total.add(a.bias, delta, plank);
        biases.add(a.bias, delta, plank);
        if a.neuron_type == "output" {
            output.add(a.bias, delta, plank);
        } else {
            hidden.add(a.bias, delta, plank);
        }
    }
    for (a, b) in before.synapses.iter().zip(after.synapses.iter()) {
        let delta = b.weight - a.weight;
        if delta.abs() < plank {
            continue;
        }
        total.add(a.weight, delta, plank);
        weights.add(a.weight, delta, plank);
        if output_uuids.contains(a.to_uuid.as_str()) {
            output.add(a.weight, delta, plank);
        } else {
            hidden.add(a.weight, delta, plank);
        }
    }
    Ok(UpdateStats {
        total: total.finish(),
        biases: biases.finish(),
        weights: weights.finish(),
        hidden: hidden.finish(),
        output: output.finish(),
    })
}

/// UUIDs of the creature's output neurons.
fn output_uuids(creature: &CreatureExport) -> std::collections::HashSet<&str> {
    creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect()
}

/// Aggregate budget one whole-creature update must fit inside (#109).
///
/// Every field is off by default, and an all-`None` region is the historical
/// fixed-step behaviour — the parity mode. A configured budget only ever
/// shrinks the update: the trust region caps, it never amplifies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrustRegion {
    /// Maximum L2 norm of the whole update.
    #[serde(default)]
    pub l2: Option<f64>,
    /// Maximum RMS per-gene delta.
    #[serde(default)]
    pub rms: Option<f64>,
    /// Maximum RMS *relative* parameter change (`Δ / value`).
    #[serde(default)]
    pub relative_rms: Option<f64>,
    /// Maximum L2 norm of the bias genes alone.
    #[serde(default)]
    pub bias_l2: Option<f64>,
    /// Maximum L2 norm of the weight genes alone.
    #[serde(default)]
    pub weight_l2: Option<f64>,
    /// Maximum number of genes one update may move. The largest moves are
    /// kept; the rest are held at their incumbent value.
    #[serde(default)]
    pub max_changed_genes: Option<usize>,
}

impl TrustRegion {
    /// True when at least one budget is configured.
    pub fn is_active(&self) -> bool {
        self.l2.is_some()
            || self.rms.is_some()
            || self.relative_rms.is_some()
            || self.bias_l2.is_some()
            || self.weight_l2.is_some()
            || self.max_changed_genes.is_some()
    }

    /// Refuse a budget that could not bound anything.
    ///
    /// A zero, negative or non-finite budget would either reject every update
    /// or silently do nothing; both are worse than saying so up front.
    pub fn validate(&self) -> Result<(), String> {
        for (name, budget) in [
            ("l2", self.l2),
            ("rms", self.rms),
            ("relativeRms", self.relative_rms),
            ("biasL2", self.bias_l2),
            ("weightL2", self.weight_l2),
        ] {
            if let Some(value) = budget
                && (!value.is_finite() || value <= 0.0)
            {
                return Err(format!(
                    "trust-region {name} budget must be finite and positive: {value}"
                ));
            }
        }
        if self.max_changed_genes == Some(0) {
            return Err(
                "trust-region maxChangedGenes must be at least 1 — a budget of 0 would freeze \
                 every gene"
                    .into(),
            );
        }
        Ok(())
    }

    /// Factor the proposed update must be multiplied by to fit the budget.
    ///
    /// `1.0` when the update already fits (or no rate budget is configured);
    /// never above `1.0`, so the trust region cannot turn a small step into a
    /// large one. A non-finite measured norm is refused rather than scaled:
    /// no factor bounds a `NaN`, and pretending one does would let a diverged
    /// update through as if it had been budgeted.
    pub fn scale_for(&self, stats: &UpdateStats) -> Result<f64, String> {
        let mut scale = 1.0f64;
        for (name, budget, measured) in [
            ("l2", self.l2, stats.total.l2),
            ("rms", self.rms, stats.total.rms),
            (
                "relativeRms",
                self.relative_rms,
                stats.total.relative_rms.unwrap_or(0.0),
            ),
            ("biasL2", self.bias_l2, stats.biases.l2),
            ("weightL2", self.weight_l2, stats.weights.l2),
        ] {
            let Some(budget) = budget else {
                continue;
            };
            if !measured.is_finite() {
                return Err(format!(
                    "trust-region {name} budget cannot bound a non-finite update norm ({measured})"
                ));
            }
            if measured > budget {
                scale = scale.min(budget / measured);
            }
        }
        Ok(scale)
    }
}

/// What one trust-region-limited apply did.
#[derive(Debug, Clone)]
pub struct TrustRegionApply {
    /// The candidate creature, after any rescale and gene-count trim.
    pub candidate: CreatureExport,
    /// The step scale the caller asked for, as the applier resolved it.
    pub requested_step_scale: f64,
    /// The step scale actually applied — `requested × scale`.
    pub realised_step_scale: f64,
    /// Factor the proposed update was multiplied by (`1.0` = untouched).
    pub scale: f64,
    /// Norms of the proposal at the requested step, measured before apply.
    pub proposed: UpdateStats,
    /// Norms of the update actually written to [`Self::candidate`].
    pub realised: UpdateStats,
    /// Genes held at their incumbent value by the changed-gene budget.
    pub trimmed_genes: usize,
}

/// Apply accumulated learning, rescaled so the whole-creature update fits
/// `region` (#109).
///
/// With an empty region this is exactly [`apply_learnings_with`] plus the
/// measurement, so the fixed-step parity mode costs one extra pass over the
/// genes and changes no value.
pub fn apply_within_trust_region(
    incumbent: &CreatureExport,
    learning: &LearningSignal,
    config: &BackpropConfig,
    learning_rate: f64,
    options: ApplyOptions,
    region: TrustRegion,
) -> Result<TrustRegionApply, String> {
    region.validate()?;
    let requested = effective_step_scale(options.step_scale);
    let apply_at = |step_scale: f64| {
        apply_learnings_with(
            incumbent,
            learning,
            config,
            learning_rate,
            ApplyOptions {
                step_scale,
                ..options
            },
        )
    };

    let proposal = apply_at(requested);
    let proposed = measure_update(incumbent, &proposal, config.plank_constant)?;
    let scale = region.scale_for(&proposed)?;
    // Each delta is `step × (proposed − current)`, so rescaling the update is
    // re-applying the same learning at a smaller step — the applier stays the
    // single place a gene value is written.
    let mut candidate = if scale < 1.0 {
        apply_at(requested * scale)
    } else {
        proposal
    };
    let trimmed_genes = match region.max_changed_genes {
        Some(max) => trim_to_gene_budget(incumbent, &mut candidate, config.plank_constant, max),
        None => 0,
    };
    let realised = measure_update(incumbent, &candidate, config.plank_constant)?;
    Ok(TrustRegionApply {
        candidate,
        requested_step_scale: requested,
        realised_step_scale: requested * scale,
        scale,
        proposed,
        realised,
        trimmed_genes,
    })
}

/// One moved gene, for the changed-gene budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gene {
    /// Neuron bias at this export index.
    Bias(usize),
    /// Synapse weight at this export index.
    Weight(usize),
}

/// Hold every gene outside the `max` largest moves at its incumbent value.
///
/// Returns how many genes were held back. Ordering is by descending `|Δ|`,
/// with biases before weights and lower export index first on a tie, so the
/// same proposal always trims to the same candidate.
fn trim_to_gene_budget(
    incumbent: &CreatureExport,
    candidate: &mut CreatureExport,
    plank: f64,
    max: usize,
) -> usize {
    let mut moved: Vec<(f64, Gene)> = Vec::new();
    for (i, (a, b)) in incumbent
        .neurons
        .iter()
        .zip(candidate.neurons.iter())
        .enumerate()
    {
        let delta = (b.bias - a.bias).abs();
        if delta >= plank {
            moved.push((delta, Gene::Bias(i)));
        }
    }
    for (i, (a, b)) in incumbent
        .synapses
        .iter()
        .zip(candidate.synapses.iter())
        .enumerate()
    {
        let delta = (b.weight - a.weight).abs();
        if delta >= plank {
            moved.push((delta, Gene::Weight(i)));
        }
    }
    if moved.len() <= max {
        return 0;
    }
    moved.sort_by(|(left, left_gene), (right, right_gene)| {
        right
            .partial_cmp(left)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| gene_key(*left_gene).cmp(&gene_key(*right_gene)))
    });
    let trimmed = moved.len() - max;
    for (_, gene) in &moved[max..] {
        match gene {
            Gene::Bias(i) => candidate.neurons[*i].bias = incumbent.neurons[*i].bias,
            Gene::Weight(i) => candidate.synapses[*i].weight = incumbent.synapses[*i].weight,
        }
    }
    trimmed
}

/// Deterministic tie-break key: biases before weights, then export order.
fn gene_key(gene: Gene) -> (u8, usize) {
    match gene {
        Gene::Bias(i) => (0, i),
        Gene::Weight(i) => (1, i),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;

    const CHAIN: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
      "neurons":[
        {"type":"hidden","uuid":"h1","bias":1.0,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":2.0,"squash":"IDENTITY"}
      ],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"h1","weight":4.0},
        {"fromUUID":"h1","toUUID":"o1","weight":8.0}
      ]
    }"#;

    /// The chain with every gene moved by `delta`.
    fn moved_by(delta: f64) -> (CreatureExport, CreatureExport) {
        let before = parse_creature_json(CHAIN).unwrap();
        let mut after = before.clone();
        for neuron in &mut after.neurons {
            neuron.bias += delta;
        }
        for synapse in &mut after.synapses {
            synapse.weight += delta;
        }
        (before, after)
    }

    #[test]
    fn norms_measure_the_genes_that_actually_moved() {
        let (before, after) = moved_by(0.1);
        let stats = measure_update(&before, &after, 1e-7).unwrap();
        assert_eq!(stats.total.changed, 4);
        assert!((stats.total.l1 - 0.4).abs() < 1e-12);
        assert!((stats.total.l2 - (4.0f64 * 0.01).sqrt()).abs() < 1e-12);
        assert!((stats.total.rms - 0.1).abs() < 1e-12, "{}", stats.total.rms);
        assert!((stats.total.max_abs - 0.1).abs() < 1e-12);
        // Incumbent values 1, 2, 4, 8 → ||θ|| = sqrt(85).
        let relative = stats.total.relative_l2.expect("values are non-zero");
        assert!((relative - stats.total.l2 / 85.0f64.sqrt()).abs() < 1e-12);
        // Per-gene relative moves 0.1/1, 0.1/2, 0.1/4, 0.1/8.
        let rms_relative = stats.total.relative_rms.expect("values clear the plank");
        let expected = ((0.1f64 / 1.0).powi(2)
            + (0.1f64 / 2.0).powi(2)
            + (0.1f64 / 4.0).powi(2)
            + (0.1f64 / 8.0).powi(2))
            / 4.0;
        assert!((rms_relative - expected.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn classes_partition_the_same_genes() {
        let (before, after) = moved_by(0.1);
        let stats = measure_update(&before, &after, 1e-7).unwrap();
        assert_eq!(stats.biases.changed + stats.weights.changed, 4);
        assert_eq!(stats.hidden.changed + stats.output.changed, 4);
        assert_eq!(stats.biases.changed, 2);
        assert_eq!(stats.weights.changed, 2);
        // h1's bias and the input→h1 weight are hidden; o1's bias and the
        // h1→o1 weight target the output.
        assert_eq!(stats.hidden.changed, 2);
        assert_eq!(stats.output.changed, 2);
        assert!((stats.biases.l1 - 0.2).abs() < 1e-12);
        assert!((stats.output.l1 - 0.2).abs() < 1e-12);
    }

    #[test]
    fn a_move_below_the_plank_is_not_an_update() {
        let (before, after) = moved_by(1e-9);
        let stats = measure_update(&before, &after, 1e-7).unwrap();
        assert_eq!(stats.total.changed, 0);
        assert_eq!(stats.total.l2, 0.0);
        assert_eq!(stats.total.rms, 0.0);
        assert_eq!(stats.total.relative_l2, None);
        assert_eq!(stats.total.relative_rms, None);
    }

    #[test]
    fn a_shape_mismatch_fails_loudly() {
        let (before, mut after) = moved_by(0.1);
        after.synapses.pop();
        let err = measure_update(&before, &after, 1e-7).unwrap_err();
        assert!(err.contains("different shape"), "{err}");
    }

    #[test]
    fn an_unconfigured_region_never_rescales() {
        let (before, after) = moved_by(0.1);
        let stats = measure_update(&before, &after, 1e-7).unwrap();
        let region = TrustRegion::default();
        assert!(!region.is_active());
        assert_eq!(region.scale_for(&stats).unwrap(), 1.0);
    }

    #[test]
    fn the_tightest_budget_decides_the_scale() {
        let (before, after) = moved_by(0.1);
        let stats = measure_update(&before, &after, 1e-7).unwrap();
        // L2 is 0.2; a budget of 0.05 asks for a quarter of the update.
        let region = TrustRegion {
            l2: Some(0.05),
            ..TrustRegion::default()
        };
        assert!((region.scale_for(&stats).unwrap() - 0.25).abs() < 1e-12);
        // A second, tighter budget wins: bias L2 is sqrt(0.02) ≈ 0.1414.
        let region = TrustRegion {
            l2: Some(0.05),
            bias_l2: Some(0.0141421356237309),
            ..TrustRegion::default()
        };
        assert!(region.scale_for(&stats).unwrap() < 0.11);
        // A budget the update already fits never grows it.
        let region = TrustRegion {
            l2: Some(100.0),
            ..TrustRegion::default()
        };
        assert_eq!(region.scale_for(&stats).unwrap(), 1.0);
    }

    #[test]
    fn a_non_finite_update_is_refused_not_scaled() {
        let (before, mut after) = moved_by(0.1);
        after.neurons[0].bias = f64::NAN;
        let stats = measure_update(&before, &after, 1e-7).unwrap();
        let region = TrustRegion {
            l2: Some(0.05),
            ..TrustRegion::default()
        };
        let err = region.scale_for(&stats).unwrap_err();
        assert!(err.contains("non-finite"), "{err}");
    }

    #[test]
    fn an_unusable_budget_is_refused() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let region = TrustRegion {
                l2: Some(bad),
                ..TrustRegion::default()
            };
            assert!(region.validate().is_err(), "{bad}");
        }
        let region = TrustRegion {
            max_changed_genes: Some(0),
            ..TrustRegion::default()
        };
        assert!(region.validate().unwrap_err().contains("maxChangedGenes"));
        assert!(TrustRegion::default().validate().is_ok());
    }

    #[test]
    fn the_gene_budget_keeps_the_largest_moves() {
        let before = parse_creature_json(CHAIN).unwrap();
        let mut candidate = before.clone();
        candidate.neurons[0].bias += 0.5; // largest
        candidate.neurons[1].bias += 0.01; // smallest
        candidate.synapses[0].weight += 0.2;
        candidate.synapses[1].weight += 0.05;
        let trimmed = trim_to_gene_budget(&before, &mut candidate, 1e-7, 2);
        assert_eq!(trimmed, 2);
        assert!((candidate.neurons[0].bias - 1.5).abs() < 1e-12, "kept");
        assert!((candidate.synapses[0].weight - 4.2).abs() < 1e-12, "kept");
        assert!(
            (candidate.neurons[1].bias - before.neurons[1].bias).abs() < 1e-12,
            "the smallest bias move is held"
        );
        assert!(
            (candidate.synapses[1].weight - before.synapses[1].weight).abs() < 1e-12,
            "the smallest weight move is held"
        );
        // A budget above the moved-gene count changes nothing.
        let mut untouched = before.clone();
        untouched.neurons[0].bias += 0.5;
        let unchanged = untouched.clone();
        assert_eq!(trim_to_gene_budget(&before, &mut untouched, 1e-7, 8), 0);
        assert_eq!(untouched.neurons[0].bias, unchanged.neurons[0].bias);
    }

    /// Ties must not depend on iteration order — the same proposal has to trim
    /// to the same candidate on every run.
    #[test]
    fn equal_moves_trim_deterministically() {
        let before = parse_creature_json(CHAIN).unwrap();
        let mut first = before.clone();
        for neuron in &mut first.neurons {
            neuron.bias += 0.25;
        }
        for synapse in &mut first.synapses {
            synapse.weight += 0.25;
        }
        let mut second = first.clone();
        assert_eq!(trim_to_gene_budget(&before, &mut first, 1e-7, 2), 2);
        assert_eq!(trim_to_gene_budget(&before, &mut second, 1e-7, 2), 2);
        assert_eq!(first.neurons[0].bias, second.neurons[0].bias);
        assert_eq!(first.synapses[1].weight, second.synapses[1].weight);
        // Biases sort ahead of weights on an exact tie.
        assert!((first.neurons[1].bias - 2.25).abs() < 1e-12);
        assert!((first.synapses[0].weight - 4.0).abs() < 1e-12);
    }
}
