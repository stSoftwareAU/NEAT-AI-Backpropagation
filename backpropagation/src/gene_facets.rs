//! Gene-class facets for the gradient diagnostics (issue #107).
//!
//! A NEAT creature is heterogeneous: different squashes, aggregate neurons,
//! deep paths, saturated units and wildly different fan-in all sit in the same
//! graph, so one whole-creature sign-agreement percentage says nothing about
//! *where* a backprop proposal is trustworthy. This module turns each probed
//! gene into a set of `(facet, bucket)` labels — squash, aggregate vs
//! ordinary, depth, fan-in / fan-out, activation health, proposal magnitude —
//! and aggregates the probe rows by every one of them.
//!
//! It owns no maths of its own: [`crate::gradient_check`] measures the genes
//! and hands the outcome here as [`FacetRow`]s, which keeps the labelling and
//! ranking independently testable.

use neat_core::{CreatureExport, SquashType, apply_get_range, parse_squash_name};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::VecDeque;

use crate::propagate_layout::NeuronTraceStats;

/// A squash-range endpoint beyond this magnitude is neat-core's unbounded
/// sentinel (`F32_LARGE`), not a real saturation ceiling.
const RANGE_BOUND: f64 = 1e6;
/// Fraction of a two-sided squash range that counts as saturated at each end.
const SATURATION_FRACTION: f64 = 0.02;
/// Distance from a one-sided range endpoint that counts as saturated.
const SATURATION_ABS: f64 = 1e-3;
/// Mean activation / activation spread below which a neuron is low-activity.
const LOW_ACTIVITY: f64 = 1e-6;

/// Activation health of the neuron a gene belongs to, read from the
/// accumulate trace.
pub mod activity {
    /// The accumulate pass never activated the neuron.
    pub const UNOBSERVED: &str = "unobserved";
    /// Mean activation sits at an end of the squash's own range.
    pub const SATURATED: &str = "saturated";
    /// The neuron barely moves — a near-zero or near-constant activation.
    pub const LOW_ACTIVITY: &str = "lowActivity";
    /// Neither saturated nor flat.
    pub const ACTIVE: &str = "active";
}

/// Facet names — the axes the probe rows are stratified by.
pub mod facet {
    /// Bias vs weight.
    pub const GENE_KIND: &str = "geneKind";
    /// Output vs hidden.
    pub const ROLE: &str = "role";
    /// The four `GeneClass` labels (kind × role).
    pub const CLASS: &str = "class";
    /// Squash name of the gene's neuron.
    pub const SQUASH: &str = "squash";
    /// Aggregate vs ordinary neuron.
    pub const AGGREGATE: &str = "aggregate";
    /// Longest-path depth bucket.
    pub const DEPTH: &str = "depth";
    /// Fan-in bucket.
    pub const FAN_IN: &str = "fanIn";
    /// Fan-out bucket.
    pub const FAN_OUT: &str = "fanOut";
    /// Activation-health bucket.
    pub const ACTIVITY: &str = "activity";
    /// Proposal magnitude decade bucket.
    pub const PROPOSAL_MAGNITUDE: &str = "proposalMagnitude";
}

/// Longest-path depth and degree of every non-input neuron.
///
/// Indexed by position in [`CreatureExport::neurons`] — the same index the
/// probe rows carry for a bias gene.
#[derive(Debug, Clone)]
pub struct CreatureTopology {
    depth: Vec<usize>,
    fan_in: Vec<usize>,
    fan_out: Vec<usize>,
    index_of: HashMap<String, usize>,
}

impl CreatureTopology {
    /// Measure `creature`'s non-input neurons.
    ///
    /// Depth is the longest hop count from an input (or from a source neuron
    /// with no inward synapse, which is depth `0`); it is resolved by a Kahn
    /// sweep rather than recursion so a production-size creature cannot blow
    /// the stack. A neuron left unresolved by the sweep — only reachable in a
    /// cycle, which the forward-only loader already refuses — keeps its
    /// one-hop base depth instead of looping.
    pub fn of(creature: &CreatureExport) -> Self {
        let count = creature.neurons.len();
        let mut index_of: HashMap<String, usize> = HashMap::with_capacity(count);
        for (i, neuron) in creature.neurons.iter().enumerate() {
            index_of.insert(neuron.uuid.clone(), i);
        }

        let mut fan_in = vec![0usize; count];
        let mut fan_out = vec![0usize; count];
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); count];
        let mut pending: Vec<usize> = vec![0; count];
        for syn in &creature.synapses {
            let to = index_of.get(&syn.to_uuid).copied();
            let from = index_of.get(&syn.from_uuid).copied();
            if let Some(to) = to {
                fan_in[to] += 1;
            }
            if let Some(from) = from {
                fan_out[from] += 1;
            }
            if let (Some(from), Some(to)) = (from, to)
                && from != to
            {
                successors[from].push(to);
                pending[to] += 1;
            }
        }

        // Depth 1 for a neuron fed by anything, 0 for a source neuron.
        let mut depth: Vec<usize> = (0..count)
            .map(|i| usize::from(fan_in[i] > 0))
            .collect::<Vec<_>>();
        let mut queue: VecDeque<usize> = (0..count).filter(|&i| pending[i] == 0).collect();
        while let Some(node) = queue.pop_front() {
            for &next in &successors[node] {
                depth[next] = depth[next].max(depth[node] + 1);
                pending[next] -= 1;
                if pending[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        Self {
            depth,
            fan_in,
            fan_out,
            index_of,
        }
    }

    /// Position of `uuid` in [`CreatureExport::neurons`], or `None` for an
    /// input neuron / unknown UUID.
    pub fn neuron_index(&self, uuid: &str) -> Option<usize> {
        self.index_of.get(uuid).copied()
    }

    /// Longest-path depth of neuron `index`.
    pub fn depth(&self, index: usize) -> usize {
        self.depth.get(index).copied().unwrap_or_default()
    }

    /// Inward synapse count of neuron `index`.
    pub fn fan_in(&self, index: usize) -> usize {
        self.fan_in.get(index).copied().unwrap_or_default()
    }

    /// Outward synapse count of neuron `index`.
    pub fn fan_out(&self, index: usize) -> usize {
        self.fan_out.get(index).copied().unwrap_or_default()
    }
}

/// Structural and activation attributes of the neuron a gene belongs to.
///
/// A weight gene takes the attributes of the neuron it *targets* — that is the
/// unit whose local topology and saturation decide whether the weight's
/// proposal is trustworthy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GeneAttributes {
    /// Squash name as exported (`IDENTITY` when the creature omits it).
    pub squash: String,
    /// True for the aggregate squashes (`MINIMUM`, `MAXIMUM`, `IF`, `HYPOT`,
    /// `HYPOTv2`, `MEAN`).
    pub aggregate: bool,
    /// Longest hop count from an input.
    pub depth: usize,
    /// Inward synapse count.
    pub fan_in: usize,
    /// Outward synapse count.
    pub fan_out: usize,
    /// Activation-health bucket — see [`activity`].
    pub activity: String,
}

/// Attributes of the neuron at `index`, using `traces` for activation health.
///
/// Fails loudly on a squash name neat-core cannot parse rather than defaulting
/// the neuron to `IDENTITY` and reporting a facet that is not true.
pub fn attributes_for(
    creature: &CreatureExport,
    topology: &CreatureTopology,
    traces: &[NeuronTraceStats],
    index: usize,
) -> Result<GeneAttributes, String> {
    let neuron = creature
        .neurons
        .get(index)
        .ok_or_else(|| format!("neuron index {index} is outside the creature"))?;
    let name = neuron.squash.as_deref().unwrap_or("IDENTITY");
    let squash = parse_squash_name(name).map_err(|e| e.to_string())?;
    Ok(GeneAttributes {
        squash: name.to_string(),
        aggregate: squash.is_aggregate(),
        depth: topology.depth(index),
        fan_in: topology.fan_in(index),
        fan_out: topology.fan_out(index),
        activity: activity_bucket(traces.get(index), squash).to_string(),
    })
}

/// Classify a neuron's activation health from its accumulate trace.
///
/// The spread test needs at least two records — with one record every neuron
/// has a zero spread and would read as flat.
pub fn activity_bucket(stats: Option<&NeuronTraceStats>, squash: SquashType) -> &'static str {
    let Some(stats) = stats.filter(|s| s.records > 0) else {
        return activity::UNOBSERVED;
    };
    let mean = stats.total_activation / stats.records as f64;
    if is_saturated(mean, squash) {
        return activity::SATURATED;
    }
    let spread = stats.maximum_activation - stats.minimum_activation;
    if mean.abs() < LOW_ACTIVITY || (stats.records >= 2 && spread < LOW_ACTIVITY) {
        return activity::LOW_ACTIVITY;
    }
    activity::ACTIVE
}

/// True when `mean` sits at an end of the squash's own output range.
fn is_saturated(mean: f64, squash: SquashType) -> bool {
    let (low, high) = apply_get_range(squash);
    let low = bounded(f64::from(low));
    let high = bounded(f64::from(high));
    match (low, high) {
        (Some(low), Some(high)) if high > low => {
            let position = (mean - low) / (high - low);
            position <= SATURATION_FRACTION || position >= 1.0 - SATURATION_FRACTION
        }
        (Some(low), None) => (mean - low).abs() <= SATURATION_ABS,
        (None, Some(high)) => (high - mean).abs() <= SATURATION_ABS,
        _ => false,
    }
}

/// `Some(value)` for a real range endpoint, `None` for the unbounded sentinel.
fn bounded(value: f64) -> Option<f64> {
    (value.is_finite() && value.abs() < RANGE_BOUND).then_some(value)
}

/// Bucket a longest-path depth.
pub fn depth_bucket(depth: usize) -> &'static str {
    match depth {
        0..=1 => "0-1",
        2..=3 => "2-3",
        4..=7 => "4-7",
        8..=15 => "8-15",
        _ => "16+",
    }
}

/// Bucket a fan-in / fan-out degree.
pub fn degree_bucket(degree: usize) -> &'static str {
    match degree {
        0 => "0",
        1 => "1",
        2..=3 => "2-3",
        4..=7 => "4-7",
        8..=15 => "8-15",
        _ => "16+",
    }
}

/// Bucket a proposal delta by decade of magnitude.
pub fn magnitude_bucket(delta: f64) -> &'static str {
    let magnitude = delta.abs();
    if !magnitude.is_finite() {
        return "nonFinite";
    }
    if magnitude < 1e-6 {
        "<1e-6"
    } else if magnitude < 1e-4 {
        "1e-6..1e-4"
    } else if magnitude < 1e-2 {
        "1e-4..1e-2"
    } else {
        ">=1e-2"
    }
}

/// One probed gene, reduced to what the facet aggregation needs.
#[derive(Debug, Clone, Copy)]
pub struct FacetRow<'a> {
    /// Gene class label (`hiddenBias`, `outputWeight`, …).
    pub class: &'a str,
    /// `bias` or `weight`.
    pub gene_kind: &'a str,
    /// `output` or `hidden`.
    pub role: &'a str,
    /// Neuron attributes — squash, aggregate flag, depth, degrees, activity.
    pub attributes: &'a GeneAttributes,
    /// Proposal delta after step scale.
    pub proposal_delta: f64,
    /// True when the finite difference cleared the floor.
    pub scored: bool,
    /// True when the proposal points downhill on the FD gradient.
    pub sign_agree: bool,
    /// True when applying the proposal actually lowered slice MSE.
    pub improved: bool,
    /// `|proposal| / |fd|`.
    pub magnitude_ratio: Option<f64>,
    /// Relative error of the first-order MSE prediction.
    pub rel_error: Option<f64>,
}

/// Aggregate outcome for one `(facet, bucket)` pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FacetStats {
    /// Facet name — see [`facet`].
    pub facet: String,
    /// Bucket within the facet.
    pub bucket: String,
    /// Genes sampled in this bucket.
    pub sampled: usize,
    /// Genes whose finite difference cleared the floor.
    pub scored: usize,
    /// Scored genes whose proposal pointed downhill.
    pub sign_agree: usize,
    /// `sign_agree / scored` as a percentage (0 when nothing scored).
    pub sign_agree_pct: f64,
    /// Sampled genes whose proposal actually lowered slice MSE.
    pub improved: usize,
    /// `improved / sampled` as a percentage.
    pub improved_pct: f64,
    /// Median `|proposal| / |fd|`.
    pub magnitude_ratio_p50: Option<f64>,
    /// Median relative error of the first-order MSE prediction.
    pub rel_error_p50: Option<f64>,
    /// 90th percentile of that relative error.
    pub rel_error_p90: Option<f64>,
}

/// Label every facet bucket `row` belongs to.
pub fn facets_of(row: &FacetRow<'_>) -> Vec<(&'static str, String)> {
    vec![
        (facet::GENE_KIND, row.gene_kind.to_string()),
        (facet::ROLE, row.role.to_string()),
        (facet::CLASS, row.class.to_string()),
        (facet::SQUASH, row.attributes.squash.clone()),
        (
            facet::AGGREGATE,
            if row.attributes.aggregate {
                "aggregate".to_string()
            } else {
                "ordinary".to_string()
            },
        ),
        (facet::DEPTH, depth_bucket(row.attributes.depth).to_string()),
        (
            facet::FAN_IN,
            degree_bucket(row.attributes.fan_in).to_string(),
        ),
        (
            facet::FAN_OUT,
            degree_bucket(row.attributes.fan_out).to_string(),
        ),
        (facet::ACTIVITY, row.attributes.activity.clone()),
        (
            facet::PROPOSAL_MAGNITUDE,
            magnitude_bucket(row.proposal_delta).to_string(),
        ),
    ]
}

/// Aggregate every probe row across every facet.
///
/// Output order is deterministic: facets in [`facets_of`] order, buckets
/// sorted by name, so two runs of the same seed produce byte-identical JSON.
pub fn aggregate_facets(rows: &[FacetRow<'_>]) -> Vec<FacetStats> {
    let mut order: Vec<&'static str> = Vec::new();
    let mut buckets: HashMap<(&'static str, String), Vec<usize>> = HashMap::new();
    for (i, row) in rows.iter().enumerate() {
        for (facet, bucket) in facets_of(row) {
            if !order.contains(&facet) {
                order.push(facet);
            }
            buckets.entry((facet, bucket)).or_default().push(i);
        }
    }

    let mut stats = Vec::new();
    for facet in order {
        let mut names: Vec<&String> = buckets
            .keys()
            .filter(|(f, _)| *f == facet)
            .map(|(_, b)| b)
            .collect();
        names.sort();
        for name in names {
            let members = &buckets[&(facet, name.clone())];
            stats.push(bucket_stats(facet, name, members, rows));
        }
    }
    stats
}

/// Fold one bucket's member rows into a [`FacetStats`].
fn bucket_stats(facet: &str, bucket: &str, members: &[usize], rows: &[FacetRow<'_>]) -> FacetStats {
    let sampled = members.len();
    let scored_rows: Vec<&FacetRow<'_>> = members
        .iter()
        .map(|&i| &rows[i])
        .filter(|r| r.scored)
        .collect();
    let scored = scored_rows.len();
    let sign_agree = scored_rows.iter().filter(|r| r.sign_agree).count();
    let improved = members.iter().filter(|&&i| rows[i].improved).count();
    let ratios = sorted_values(scored_rows.iter().filter_map(|r| r.magnitude_ratio));
    let errors = sorted_values(members.iter().filter_map(|&i| rows[i].rel_error));
    FacetStats {
        facet: facet.to_string(),
        bucket: bucket.to_string(),
        sampled,
        scored,
        sign_agree,
        sign_agree_pct: percentage(sign_agree, scored),
        improved,
        improved_pct: percentage(improved, sampled),
        magnitude_ratio_p50: percentile(&ratios, 0.50),
        rel_error_p50: percentile(&errors, 0.50),
        rel_error_p90: percentile(&errors, 0.90),
    }
}

/// Collect finite values in ascending order.
fn sorted_values(values: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut out: Vec<f64> = values.filter(|v| v.is_finite()).collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// `part / whole` as a percentage, 0 when `whole` is 0.
fn percentage(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

/// Nearest-rank percentile of an ascending slice.
pub fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    Some(sorted[idx.min(sorted.len() - 1)])
}

/// Best and worst `(facet, bucket)` pairs by sign agreement.
///
/// Only buckets with at least `min_scored` scored genes are ranked — a bucket
/// of two genes says nothing — and a facet with a single bucket is skipped
/// entirely because it offers no contrast. Ties break on relative error, then
/// on facet and bucket name, so the ranking is stable across runs.
///
/// Both lists are views of one ranking: with fewer than `2 × limit` eligible
/// buckets they overlap, which is the honest reading of a small sample rather
/// than two independent findings.
pub fn rank_facets(
    stats: &[FacetStats],
    min_scored: usize,
    limit: usize,
) -> (Vec<FacetStats>, Vec<FacetStats>) {
    let mut eligible: Vec<FacetStats> = stats
        .iter()
        .filter(|s| s.scored >= min_scored.max(1))
        .filter(|s| stats.iter().filter(|o| o.facet == s.facet).count() > 1)
        .cloned()
        .collect();
    eligible.sort_by(|a, b| {
        b.sign_agree_pct
            .partial_cmp(&a.sign_agree_pct)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                error_key(a)
                    .partial_cmp(&error_key(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.facet.cmp(&b.facet))
            .then_with(|| a.bucket.cmp(&b.bucket))
    });
    let best: Vec<FacetStats> = eligible.iter().take(limit).cloned().collect();
    let worst: Vec<FacetStats> = eligible.iter().rev().take(limit).cloned().collect();
    (best, worst)
}

/// Sort key for the relative-error tie-break — a missing error sorts last.
fn error_key(stats: &FacetStats) -> f64 {
    stats.rel_error_p50.unwrap_or(f64::INFINITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;

    const CHAIN: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
      "neurons":[
        {"type":"hidden","uuid":"h1","bias":0.0,"squash":"TANH"},
        {"type":"hidden","uuid":"h2","bias":0.0,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"MEAN"}
      ],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"input-1","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"h2","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0},
        {"fromUUID":"h2","toUUID":"o1","weight":1.0}
      ]
    }"#;

    fn attributes(index: usize, traces: &[NeuronTraceStats]) -> GeneAttributes {
        let creature = parse_creature_json(CHAIN).unwrap();
        let topology = CreatureTopology::of(&creature);
        attributes_for(&creature, &topology, traces, index).unwrap()
    }

    #[test]
    fn depth_is_the_longest_path_not_the_shortest() {
        let creature = parse_creature_json(CHAIN).unwrap();
        let topology = CreatureTopology::of(&creature);
        assert_eq!(topology.depth(0), 1, "h1 is fed by inputs only");
        assert_eq!(topology.depth(1), 2, "h2 sits behind h1");
        // o1 is reachable in one hop from h1 and two from h2 — the longer wins.
        assert_eq!(topology.depth(2), 3, "o1 must take the longest path");
    }

    #[test]
    fn degrees_count_input_edges_too() {
        let creature = parse_creature_json(CHAIN).unwrap();
        let topology = CreatureTopology::of(&creature);
        assert_eq!(topology.fan_in(0), 2, "h1 takes both inputs");
        assert_eq!(topology.fan_out(0), 2, "h1 feeds h2 and o1");
        assert_eq!(topology.fan_in(2), 2, "o1 takes h1 and h2");
        assert_eq!(topology.fan_out(2), 0);
    }

    #[test]
    fn unknown_neuron_index_fails_loudly() {
        let creature = parse_creature_json(CHAIN).unwrap();
        let topology = CreatureTopology::of(&creature);
        let err = attributes_for(&creature, &topology, &[], 99).expect_err("index is out of range");
        assert!(err.contains("outside the creature"), "{err}");
    }

    #[test]
    fn aggregate_neurons_are_labelled() {
        assert!(attributes(2, &[]).aggregate, "MEAN is an aggregate squash");
        assert!(!attributes(0, &[]).aggregate, "TANH is not");
        assert_eq!(attributes(0, &[]).squash, "TANH");
    }

    /// Build a trace with `records` observations of a constant activation.
    fn trace(records: u64, activation: f64) -> NeuronTraceStats {
        NeuronTraceStats {
            records,
            total_activation: activation * records as f64,
            maximum_activation: activation,
            minimum_activation: activation,
            ..NeuronTraceStats::default()
        }
    }

    #[test]
    fn activity_buckets_split_saturated_flat_and_active() {
        assert_eq!(
            activity_bucket(None, SquashType::Tanh),
            activity::UNOBSERVED
        );
        assert_eq!(
            activity_bucket(Some(&trace(0, 0.0)), SquashType::Tanh),
            activity::UNOBSERVED
        );
        assert_eq!(
            activity_bucket(Some(&trace(8, 0.999)), SquashType::Tanh),
            activity::SATURATED,
            "a TANH pinned at +1 is saturated"
        );
        assert_eq!(
            activity_bucket(Some(&trace(8, 0.0)), SquashType::Identity),
            activity::LOW_ACTIVITY,
            "an unbounded neuron sitting at zero is low-activity"
        );
        let mut moving = trace(8, 0.5);
        moving.minimum_activation = 0.1;
        moving.maximum_activation = 0.9;
        assert_eq!(
            activity_bucket(Some(&moving), SquashType::Tanh),
            activity::ACTIVE
        );
    }

    #[test]
    fn one_record_is_not_reported_as_flat() {
        let mut single = trace(1, 0.5);
        single.minimum_activation = 0.5;
        single.maximum_activation = 0.5;
        assert_eq!(
            activity_bucket(Some(&single), SquashType::Tanh),
            activity::ACTIVE,
            "a single record has no spread to judge"
        );
    }

    #[test]
    fn buckets_cover_their_boundaries() {
        assert_eq!(depth_bucket(0), "0-1");
        assert_eq!(depth_bucket(1), "0-1");
        assert_eq!(depth_bucket(2), "2-3");
        assert_eq!(depth_bucket(16), "16+");
        assert_eq!(degree_bucket(0), "0");
        assert_eq!(degree_bucket(3), "2-3");
        assert_eq!(degree_bucket(99), "16+");
        assert_eq!(magnitude_bucket(0.0), "<1e-6");
        assert_eq!(magnitude_bucket(-1e-5), "1e-6..1e-4");
        assert_eq!(magnitude_bucket(0.5), ">=1e-2");
        assert_eq!(magnitude_bucket(f64::NAN), "nonFinite");
    }

    fn row<'a>(attributes: &'a GeneAttributes, sign_agree: bool, improved: bool) -> FacetRow<'a> {
        FacetRow {
            class: "hiddenBias",
            gene_kind: "bias",
            role: "hidden",
            attributes,
            proposal_delta: 0.05,
            scored: true,
            sign_agree,
            improved,
            magnitude_ratio: Some(2.0),
            rel_error: Some(if sign_agree { 0.1 } else { 0.9 }),
        }
    }

    #[test]
    fn aggregation_counts_every_row_in_every_facet() {
        let good = attributes(0, &[]);
        let bad = attributes(2, &[]);
        let rows = vec![
            row(&good, true, true),
            row(&good, true, true),
            row(&bad, false, false),
        ];
        let stats = aggregate_facets(&rows);
        let aggregate_facet: Vec<&FacetStats> = stats
            .iter()
            .filter(|s| s.facet == facet::AGGREGATE)
            .collect();
        assert_eq!(aggregate_facet.len(), 2, "ordinary and aggregate buckets");
        let sampled: usize = aggregate_facet.iter().map(|s| s.sampled).sum();
        assert_eq!(sampled, rows.len(), "every row lands in exactly one bucket");
        let ordinary = aggregate_facet
            .iter()
            .find(|s| s.bucket == "ordinary")
            .unwrap();
        assert_eq!(ordinary.sampled, 2);
        assert_eq!(ordinary.sign_agree, 2);
        assert!((ordinary.sign_agree_pct - 100.0).abs() < 1e-9);
        assert!((ordinary.improved_pct - 100.0).abs() < 1e-9);
        assert_eq!(ordinary.rel_error_p50, Some(0.1));
    }

    #[test]
    fn ranking_puts_the_strongest_bucket_first_and_needs_evidence() {
        let good = attributes(0, &[]);
        let bad = attributes(2, &[]);
        let mut rows = vec![row(&bad, false, false)];
        rows.extend(std::iter::repeat_n(row(&good, true, true), 4));
        let stats = aggregate_facets(&rows);

        let (best, worst) = rank_facets(&stats, 2, 3);
        assert!(!best.is_empty(), "the four-gene bucket clears the floor");
        assert!(
            best.iter().all(|s| s.scored >= 2),
            "under-sampled buckets must not be ranked"
        );
        assert!(best[0].sign_agree_pct >= worst[0].sign_agree_pct);
        assert!(
            best.iter().all(|s| s.bucket != "aggregate"),
            "the single-gene aggregate bucket has too little evidence"
        );

        // A floor above every bucket size ranks nothing at all.
        let (best, worst) = rank_facets(&stats, 99, 3);
        assert!(best.is_empty() && worst.is_empty());
    }

    #[test]
    fn ranking_skips_facets_with_a_single_bucket() {
        let good = attributes(0, &[]);
        let rows: Vec<FacetRow<'_>> = std::iter::repeat_n(row(&good, true, true), 4).collect();
        let stats = aggregate_facets(&rows);
        let (best, _) = rank_facets(&stats, 1, 20);
        assert!(
            best.is_empty(),
            "every facet has one bucket — there is nothing to contrast"
        );
    }
}
