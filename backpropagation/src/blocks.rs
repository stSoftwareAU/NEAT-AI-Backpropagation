//! Blockwise / neighbourhood gene selection (issue #105).
//!
//! One accumulation pass over the corpus produces a single
//! [`LearningSignal`]. Applying all of it moves every gene at once, and on a
//! highly evolved production creature those correlated moves destroy the
//! compensating relationships evolution found. This module carves that one
//! signal into small **blocks** — a neuron and its synapses, a bounded
//! neighbourhood, the output head, a connected subgraph, the loudest genes —
//! so many independent candidates can be generated from one expensive pass and
//! scored one at a time.
//!
//! A block never changes what a gene's proposal *is*: masking the signal keeps
//! each selected gene's accumulator verbatim, so a block candidate is exactly
//! the whole-creature candidate restricted to the block's genes.

use crate::backprop::{BackpropConfig, LearningSignal, effective_step_scale};
use crate::propagate_layout::NeuronTraceStats;
use crate::targets::{TargetPlan, TargetSelection, rank_targets, select_targets};
use neat_core::CreatureExport;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// How a block of genes was selected (issue #105).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BlockStrategy {
    /// Every gene — the whole-creature apply, kept for parity.
    Global,
    /// One neuron's bias plus every synapse incident to it.
    Neuron,
    /// One neuron plus its neighbours out to a bounded radius.
    Neighbourhood,
    /// Output neurons and the synapses that reach them.
    OutputHead,
    /// A seeded random connected subgraph.
    Subgraph,
    /// The loudest genes by accumulated proposal magnitude, wherever they sit.
    TopGenes,
}

impl BlockStrategy {
    /// Stable lower-case slug used in candidate labels and file names.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Neuron => "neuron",
            Self::Neighbourhood => "neighbourhood",
            Self::OutputHead => "output-head",
            Self::Subgraph => "subgraph",
            Self::TopGenes => "top-genes",
        }
    }
}

/// A region of a creature's genes that one candidate may move.
///
/// Indices are positions in `CreatureExport.neurons` / `CreatureExport.synapses`
/// — the same indexing [`LearningSignal`] uses.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneBlock {
    /// How this block was selected.
    pub strategy: BlockStrategy,
    /// Export neuron index the block was grown from, when it has one.
    pub focus: Option<usize>,
    /// Export neuron indices whose bias may move.
    pub neurons: BTreeSet<usize>,
    /// Export synapse indices whose weight may move.
    pub synapses: BTreeSet<usize>,
    /// Why the focus target was chosen, for a focus-grown block (issue #108).
    /// `None` for the strategies that select a region rather than a target —
    /// `global`, `output-head` and `top-genes`.
    pub selection: Option<TargetSelection>,
}

impl GeneBlock {
    /// Total genes the block may move.
    pub fn gene_count(&self) -> usize {
        self.neurons.len() + self.synapses.len()
    }

    /// Whether the block selects no gene at all.
    pub fn is_empty(&self) -> bool {
        self.gene_count() == 0
    }

    /// Restrict `signal` to this block's genes.
    ///
    /// Genes outside the block are left with a zero-count accumulator, which
    /// [`crate::backprop::apply_learnings_with`] skips — so the candidate moves
    /// the block and nothing else.
    pub fn mask(&self, signal: &LearningSignal) -> LearningSignal {
        let mut masked = LearningSignal::new(signal.biases.len(), signal.weights.len());
        for &i in &self.neurons {
            if let Some(bias) = signal.biases.get(i) {
                masked.biases[i] = bias.clone();
            }
        }
        for &i in &self.synapses {
            if let Some(weight) = signal.weights.get(i) {
                masked.weights[i] = weight.clone();
            }
        }
        masked
    }
}

/// Per-gene |Δ| the accumulated learning proposes at a given step scale.
///
/// Non-finite proposals are recorded as `0.0` so a diverged gene never ranks
/// first; the output gate ([`crate::validate::TrainedTopology`]) is what
/// refuses the creature such a gene would produce.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProposalMagnitudes {
    /// |Δbias| per export neuron index.
    pub biases: Vec<f64>,
    /// |Δweight| per export synapse index.
    pub weights: Vec<f64>,
}

/// Measure what the accumulated learning proposes for every gene, without
/// applying any of it.
pub fn proposal_magnitudes(
    creature: &CreatureExport,
    signal: &LearningSignal,
    config: &BackpropConfig,
    learning_rate: f64,
    step_scale: f64,
) -> ProposalMagnitudes {
    let step = effective_step_scale(step_scale);
    let magnitude = |proposed: f64, current: f64| {
        let delta = (proposed - current) * step;
        if delta.is_finite() { delta.abs() } else { 0.0 }
    };
    let biases = creature
        .neurons
        .iter()
        .enumerate()
        .map(|(i, neuron)| match signal.biases.get(i) {
            Some(bias) if bias.count > 0.0 => magnitude(
                bias.propose(neuron.bias, config, learning_rate),
                neuron.bias,
            ),
            _ => 0.0,
        })
        .collect();
    let weights = creature
        .synapses
        .iter()
        .enumerate()
        .map(|(i, synapse)| match signal.weights.get(i) {
            Some(weight) if weight.count > 0.0 => magnitude(
                weight.propose(synapse.weight, config, learning_rate),
                synapse.weight,
            ),
            _ => 0.0,
        })
        .collect();
    ProposalMagnitudes { biases, weights }
}

/// Adjacency of a creature's export graph.
///
/// Virtual `input-N` sources carry no bias gene, so they are neighbours of
/// nothing — their synapses still join the block of the neuron they feed.
#[derive(Debug, Clone)]
pub struct BlockGraph {
    /// Synapse indices touching each export neuron (incoming and outgoing).
    incident: Vec<Vec<usize>>,
    /// Export neuron indices adjacent to each neuron, in either direction.
    neighbours: Vec<Vec<usize>>,
}

impl BlockGraph {
    /// Build the adjacency of `creature`.
    pub fn of(creature: &CreatureExport) -> Self {
        let index: HashMap<&str, usize> = creature
            .neurons
            .iter()
            .enumerate()
            .map(|(i, neuron)| (neuron.uuid.as_str(), i))
            .collect();
        let mut incident = vec![Vec::new(); creature.neurons.len()];
        let mut neighbours = vec![Vec::new(); creature.neurons.len()];
        for (synapse_index, synapse) in creature.synapses.iter().enumerate() {
            let from = index.get(synapse.from_uuid.as_str()).copied();
            let to = index.get(synapse.to_uuid.as_str()).copied();
            if let Some(i) = from {
                incident[i].push(synapse_index);
            }
            if let Some(i) = to {
                incident[i].push(synapse_index);
            }
            if let (Some(a), Some(b)) = (from, to)
                && a != b
            {
                neighbours[a].push(b);
                neighbours[b].push(a);
            }
        }
        // A self-loop touches its neuron twice and parallel edges repeat a
        // neighbour, which would double-weight those genes in the focus
        // ranking and bias the subgraph walk towards multiply-connected pairs.
        for list in incident.iter_mut().chain(neighbours.iter_mut()) {
            list.sort_unstable();
            list.dedup();
        }
        Self {
            incident,
            neighbours,
        }
    }

    /// Synapse indices touching export neuron `index`, in either direction.
    pub fn incident(&self, index: usize) -> &[usize] {
        self.incident.get(index).map_or(&[][..], Vec::as_slice)
    }

    /// Build a block owning `neurons` and every synapse incident to them.
    pub fn block(
        &self,
        strategy: BlockStrategy,
        focus: Option<usize>,
        neurons: BTreeSet<usize>,
    ) -> GeneBlock {
        let mut synapses = BTreeSet::new();
        for &neuron in &neurons {
            synapses.extend(self.incident(neuron).iter().copied());
        }
        GeneBlock {
            strategy,
            focus,
            neurons,
            synapses,
            selection: None,
        }
    }

    /// Neurons within `radius` hops of `focus` (radius `0` is `focus` alone).
    pub fn neighbourhood(&self, focus: usize, radius: usize) -> BTreeSet<usize> {
        let mut seen = BTreeSet::from([focus]);
        let mut frontier = VecDeque::from([(focus, 0usize)]);
        while let Some((neuron, depth)) = frontier.pop_front() {
            if depth >= radius {
                continue;
            }
            for &next in self.neighbours.get(neuron).into_iter().flatten() {
                if seen.insert(next) {
                    frontier.push_back((next, depth + 1));
                }
            }
        }
        seen
    }

    /// Walk the graph from `start` collecting up to `size` connected neurons.
    ///
    /// The walk is bounded: a dead end resumes from a neuron already collected,
    /// and the whole walk gives up after a fixed budget rather than spinning on
    /// an isolated neuron.
    pub fn random_subgraph(
        &self,
        start: usize,
        size: usize,
        rng: &mut impl Rng,
    ) -> BTreeSet<usize> {
        let mut chosen = BTreeSet::from([start]);
        let mut current = start;
        let budget = size.saturating_mul(8).max(8);
        for _ in 0..budget {
            if chosen.len() >= size.max(1) {
                break;
            }
            let options = self.neighbours.get(current).map_or(&[][..], Vec::as_slice);
            if options.is_empty() {
                let members: Vec<usize> = chosen.iter().copied().collect();
                current = members[rng.random_range(0..members.len())];
                continue;
            }
            current = options[rng.random_range(0..options.len())];
            chosen.insert(current);
        }
        chosen
    }
}

/// Which blocks to generate from one accumulation pass (issue #105).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlockPlan {
    /// Strategies to generate, in order.
    pub strategies: Vec<BlockStrategy>,
    /// Focus neurons per neuron / neighbourhood / subgraph strategy.
    pub blocks_per_strategy: usize,
    /// Hops around the focus neuron for [`BlockStrategy::Neighbourhood`].
    pub radius: usize,
    /// Neurons per [`BlockStrategy::Subgraph`] walk.
    pub subgraph_size: usize,
    /// Genes kept by [`BlockStrategy::TopGenes`].
    pub top_genes: usize,
    /// How the focus targets are drawn (issue #108).
    #[serde(default)]
    pub targets: TargetPlan,
}

impl Default for BlockPlan {
    fn default() -> Self {
        Self {
            strategies: vec![
                BlockStrategy::Global,
                BlockStrategy::Neuron,
                BlockStrategy::Neighbourhood,
                BlockStrategy::OutputHead,
                BlockStrategy::Subgraph,
                BlockStrategy::TopGenes,
            ],
            blocks_per_strategy: 4,
            radius: 1,
            subgraph_size: 8,
            top_genes: 32,
            targets: TargetPlan::default(),
        }
    }
}

impl BlockPlan {
    /// Refuse a plan that could only produce nothing.
    ///
    /// A zero here is not a harmless no-op: it reads as "blockwise generation
    /// ran and found nothing" when in fact nothing was ever asked for.
    pub fn validate(&self) -> Result<(), String> {
        if self.strategies.is_empty() {
            return Err("blockwise generation needs at least one strategy".into());
        }
        if self.blocks_per_strategy == 0 {
            return Err("blocksPerStrategy must be at least 1".into());
        }
        // `radius: 0` would make every neighbourhood block identical to the
        // neuron block for the same focus, so the whole strategy would vanish
        // into the duplicate filter and read as "found nothing".
        if self.strategies.contains(&BlockStrategy::Neighbourhood) && self.radius == 0 {
            return Err(
                "radius must be at least 1 for the neighbourhood strategy — radius 0 is the \
                 neuron strategy"
                    .into(),
            );
        }
        if self.subgraph_size == 0 {
            return Err("subgraphSize must be at least 1".into());
        }
        if self.top_genes == 0 {
            return Err("topGenes must be at least 1".into());
        }
        self.targets.validate()?;
        Ok(())
    }
}

/// The neurons a focus-based strategy may grow from, in export order.
///
/// Hidden / constant neurons are the target — the output head has a strategy of
/// its own. A creature with no hidden neurons falls back to all of them, so a
/// small creature still yields blocks instead of an empty plan. Which of them
/// is worth an experiment is [`crate::targets`]' decision, not this one's.
fn focus_pool(creature: &CreatureExport) -> Vec<usize> {
    let hidden: Vec<usize> = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type != "output")
        .map(|(i, _)| i)
        .collect();
    if hidden.is_empty() {
        (0..creature.neurons.len()).collect()
    } else {
        hidden
    }
}

/// Every gene, as one block — the whole-creature apply kept for parity.
fn global_block(creature: &CreatureExport) -> GeneBlock {
    GeneBlock {
        strategy: BlockStrategy::Global,
        focus: None,
        neurons: (0..creature.neurons.len()).collect(),
        synapses: (0..creature.synapses.len()).collect(),
        selection: None,
    }
}

/// Output neurons plus the synapses that reach them.
fn output_head_block(creature: &CreatureExport, graph: &BlockGraph) -> GeneBlock {
    let outputs: BTreeSet<usize> = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, neuron)| neuron.neuron_type == "output")
        .map(|(i, _)| i)
        .collect();
    graph.block(BlockStrategy::OutputHead, None, outputs)
}

/// The `k` loudest genes by proposal magnitude, wherever they sit.
fn top_genes_block(magnitudes: &ProposalMagnitudes, k: usize) -> GeneBlock {
    // One ranking over both gene kinds: a bias and a weight compete on the
    // same |Δ|, so the block is the K loudest genes and not K of each.
    let mut genes: Vec<(bool, usize, f64)> = magnitudes
        .biases
        .iter()
        .enumerate()
        .map(|(i, &m)| (true, i, m))
        .chain(
            magnitudes
                .weights
                .iter()
                .enumerate()
                .map(|(i, &m)| (false, i, m)),
        )
        .filter(|&(_, _, m)| m > 0.0)
        .collect();
    genes.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    genes.truncate(k);
    let mut block = GeneBlock {
        strategy: BlockStrategy::TopGenes,
        focus: None,
        neurons: BTreeSet::new(),
        synapses: BTreeSet::new(),
        selection: None,
    };
    for (is_bias, index, _) in genes {
        if is_bias {
            block.neurons.insert(index);
        } else {
            block.synapses.insert(index);
        }
    }
    block
}

/// What [`plan_blocks`] produced, including what it had to drop.
///
/// The drops are counted rather than swallowed: a plan that asked for eight
/// neighbourhood blocks and got two back has to say so, or the run reads as
/// "blockwise generation found nothing" when it was the planner that discarded
/// them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BlockPlanOutcome {
    /// The blocks to generate candidates from.
    pub blocks: Vec<GeneBlock>,
    /// Blocks discarded because they selected no gene at all.
    pub dropped_empty: usize,
    /// Blocks discarded because an earlier block selected the same genes.
    pub dropped_duplicate: usize,
}

/// Generate every candidate block one accumulation pass supports.
///
/// Focus targets are drawn by [`crate::targets`] under `plan.targets`: the
/// ranked evidence the same accumulation pass produced, plus the configured
/// uniform random control share. Blocks selecting no gene, and blocks
/// selecting exactly the genes an earlier block already selected, are dropped
/// — a duplicate would cost a full scorer run to learn a number already known
/// — and both counts are reported in the returned [`BlockPlanOutcome`].
pub fn plan_blocks(
    creature: &CreatureExport,
    graph: &BlockGraph,
    magnitudes: &ProposalMagnitudes,
    signal: &LearningSignal,
    traces: &[NeuronTraceStats],
    plan: &BlockPlan,
    rng: &mut impl Rng,
) -> Result<BlockPlanOutcome, String> {
    plan.validate()?;
    let pool = focus_pool(creature);
    let ranked = rank_targets(creature, graph, magnitudes, signal, traces, &pool);
    let mut blocks = Vec::new();
    let mut dropped_empty = 0usize;
    let mut dropped_duplicate = 0usize;
    let mut seen: HashSet<(BTreeSet<usize>, BTreeSet<usize>)> = HashSet::new();
    let mut push = |block: GeneBlock, blocks: &mut Vec<GeneBlock>| {
        if block.is_empty() {
            dropped_empty += 1;
            return;
        }
        if seen.insert((block.neurons.clone(), block.synapses.clone())) {
            blocks.push(block);
        } else {
            dropped_duplicate += 1;
        }
    };
    for &strategy in &plan.strategies {
        match strategy {
            BlockStrategy::Global => push(global_block(creature), &mut blocks),
            BlockStrategy::OutputHead => push(output_head_block(creature, graph), &mut blocks),
            BlockStrategy::TopGenes => {
                push(top_genes_block(magnitudes, plan.top_genes), &mut blocks)
            }
            // The three focus strategies differ only in how far they grow
            // around the drawn target, so they share one selection loop —
            // every focus block therefore carries its selection metadata by
            // construction rather than by repetition.
            BlockStrategy::Neuron | BlockStrategy::Neighbourhood | BlockStrategy::Subgraph => {
                for target in select_targets(&plan.targets, &ranked, plan.blocks_per_strategy, rng)?
                {
                    let focus = target.neuron;
                    let neurons = match strategy {
                        BlockStrategy::Neuron => BTreeSet::from([focus]),
                        BlockStrategy::Neighbourhood => graph.neighbourhood(focus, plan.radius),
                        _ => graph.random_subgraph(focus, plan.subgraph_size, rng),
                    };
                    let mut block = graph.block(strategy, Some(focus), neurons);
                    block.selection = Some(target.selection);
                    push(block, &mut blocks);
                }
            }
        }
    }
    if blocks.is_empty() {
        return Err("blockwise generation produced no candidate blocks".into());
    }
    Ok(BlockPlanOutcome {
        blocks,
        dropped_empty,
        dropped_duplicate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backprop::{BiasSignal, WeightSignal, apply_learnings_with};
    use neat_core::parse_creature_json;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// `input-0 → h1 → h2 → o1`, plus a shortcut `h1 → o1`.
    const CHAIN: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
      "neurons":[
        {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
        {"type":"hidden","uuid":"h2","bias":0.0,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"h2","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0},
        {"fromUUID":"h2","toUUID":"o1","weight":1.0}
      ]
    }"#;

    fn chain() -> CreatureExport {
        parse_creature_json(CHAIN).unwrap()
    }

    /// A signal that wants to move every gene of [`CHAIN`].
    fn full_signal(creature: &CreatureExport) -> LearningSignal {
        let mut signal = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
        for (i, bias) in signal.biases.iter_mut().enumerate() {
            *bias = BiasSignal {
                count: 10.0,
                total_adjusted_bias: 5.0 + i as f64,
                no_change: false,
            };
        }
        for (i, weight) in signal.weights.iter_mut().enumerate() {
            *weight = WeightSignal {
                count: 4.0,
                total_positive_activation: 2.0,
                count_positive: 4.0,
                total_positive_adjusted_value: 2.0 * (2.0 + i as f64),
                ..WeightSignal::default()
            };
        }
        signal
    }

    #[test]
    fn a_neuron_block_owns_its_bias_and_every_incident_synapse() {
        let creature = chain();
        let graph = BlockGraph::of(&creature);
        // h1 (index 0) has one incoming (input-0 → h1) and two outgoing.
        let block = graph.block(BlockStrategy::Neuron, Some(0), BTreeSet::from([0]));
        assert_eq!(block.neurons, BTreeSet::from([0]));
        assert_eq!(block.synapses, BTreeSet::from([0, 1, 2]));
        assert_eq!(block.gene_count(), 4);
    }

    #[test]
    fn a_neighbourhood_grows_by_radius_and_stops() {
        let creature = chain();
        let graph = BlockGraph::of(&creature);
        assert_eq!(graph.neighbourhood(1, 0), BTreeSet::from([1]));
        // h2's neighbours are h1 and o1.
        assert_eq!(graph.neighbourhood(1, 1), BTreeSet::from([0, 1, 2]));
        // Radius beyond the graph cannot select more than the graph holds.
        assert_eq!(graph.neighbourhood(1, 9), BTreeSet::from([0, 1, 2]));
        assert_eq!(creature.neurons.len(), 3);
    }

    #[test]
    fn the_output_head_block_is_the_outputs_and_what_reaches_them() {
        let creature = chain();
        let graph = BlockGraph::of(&creature);
        let block = output_head_block(&creature, &graph);
        assert_eq!(block.strategy, BlockStrategy::OutputHead);
        assert_eq!(block.neurons, BTreeSet::from([2]));
        // h1 → o1 and h2 → o1.
        assert_eq!(block.synapses, BTreeSet::from([2, 3]));
    }

    #[test]
    fn a_random_subgraph_is_connected_and_bounded() {
        let creature = chain();
        let graph = BlockGraph::of(&creature);
        let mut rng = StdRng::seed_from_u64(11);
        let neurons = graph.random_subgraph(0, 2, &mut rng);
        assert_eq!(neurons.len(), 2);
        assert!(neurons.contains(&0));
        // Every member must be reachable from the start within the walk.
        for &member in &neurons {
            assert!(
                graph
                    .neighbourhood(0, creature.neurons.len())
                    .contains(&member)
            );
        }
        // A request larger than the graph returns the graph, not a hang.
        let all = graph.random_subgraph(0, 99, &mut rng);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn top_genes_keeps_the_loudest_genes_only() {
        let magnitudes = ProposalMagnitudes {
            biases: vec![0.5, 0.0, 0.1],
            weights: vec![0.9, 0.2, 0.0, 0.4],
        };
        let block = top_genes_block(&magnitudes, 3);
        // 0.9 (weight 0), 0.5 (bias 0), 0.4 (weight 3).
        assert_eq!(block.neurons, BTreeSet::from([0]));
        assert_eq!(block.synapses, BTreeSet::from([0, 3]));
        // Genes with no proposal never enter the block, even when K is huge.
        let everything = top_genes_block(&magnitudes, 99);
        assert_eq!(everything.gene_count(), 5);
    }

    #[test]
    fn masking_reproduces_the_global_apply_on_the_block_and_nothing_else() {
        let creature = chain();
        let signal = full_signal(&creature);
        let config = BackpropConfig::default();
        let options = crate::backprop::ApplyOptions {
            step_scale: 0.5,
            ..crate::backprop::ApplyOptions::default()
        };
        let global = apply_learnings_with(&creature, &signal, &config, 0.01, options);
        let graph = BlockGraph::of(&creature);
        let block = graph.block(BlockStrategy::Neuron, Some(0), BTreeSet::from([0]));
        let blockwise =
            apply_learnings_with(&creature, &block.mask(&signal), &config, 0.01, options);

        for (i, neuron) in blockwise.neurons.iter().enumerate() {
            if block.neurons.contains(&i) {
                assert_eq!(neuron.bias, global.neurons[i].bias, "neuron {i} in block");
                assert_ne!(neuron.bias, creature.neurons[i].bias, "neuron {i} moved");
            } else {
                assert_eq!(neuron.bias, creature.neurons[i].bias, "neuron {i} held");
            }
        }
        for (i, synapse) in blockwise.synapses.iter().enumerate() {
            if block.synapses.contains(&i) {
                assert_eq!(
                    synapse.weight, global.synapses[i].weight,
                    "synapse {i} in block"
                );
            } else {
                assert_eq!(
                    synapse.weight, creature.synapses[i].weight,
                    "synapse {i} held"
                );
            }
        }
    }

    #[test]
    fn proposal_magnitudes_measure_the_step_without_applying_it() {
        let creature = chain();
        let signal = full_signal(&creature);
        let config = BackpropConfig::default();
        let magnitudes = proposal_magnitudes(&creature, &signal, &config, 0.01, 1.0);
        assert_eq!(magnitudes.biases.len(), creature.neurons.len());
        assert_eq!(magnitudes.weights.len(), creature.synapses.len());
        assert!(magnitudes.biases.iter().all(|m| *m > 0.0));
        // Halving the step halves every proposed move.
        let half = proposal_magnitudes(&creature, &signal, &config, 0.01, 0.5);
        for (full, part) in magnitudes.biases.iter().zip(half.biases.iter()) {
            assert!((full / 2.0 - part).abs() < 1e-12);
        }
        // A gene with no accumulation proposes nothing.
        let empty = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
        let none = proposal_magnitudes(&creature, &empty, &config, 0.01, 1.0);
        assert!(none.biases.iter().all(|m| *m == 0.0));
        assert!(none.weights.iter().all(|m| *m == 0.0));
    }

    #[test]
    fn planning_yields_one_block_per_strategy_request_without_duplicates() {
        let creature = chain();
        let signal = full_signal(&creature);
        let config = BackpropConfig::default();
        let graph = BlockGraph::of(&creature);
        let magnitudes = proposal_magnitudes(&creature, &signal, &config, 0.01, 0.01);
        let mut rng = StdRng::seed_from_u64(5);
        let traces = vec![NeuronTraceStats::default(); creature.neurons.len()];
        let planned = plan_blocks(
            &creature,
            &graph,
            &magnitudes,
            &signal,
            &traces,
            &BlockPlan {
                blocks_per_strategy: 2,
                subgraph_size: 2,
                top_genes: 3,
                ..BlockPlan::default()
            },
            &mut rng,
        )
        .unwrap();
        let blocks = &planned.blocks;

        // The global block is present for parity, and it is the only block
        // holding every gene.
        let global: Vec<&GeneBlock> = blocks
            .iter()
            .filter(|b| b.strategy == BlockStrategy::Global)
            .collect();
        assert_eq!(global.len(), 1);
        assert_eq!(global[0].gene_count(), 7);
        for block in blocks
            .iter()
            .filter(|b| b.strategy != BlockStrategy::Global)
        {
            assert!(
                block.gene_count() < 7,
                "{:?} block must be smaller than the whole creature",
                block.strategy
            );
        }
        // Two focus neurons per focus-based strategy, both hidden.
        let neuron_blocks: Vec<&GeneBlock> = blocks
            .iter()
            .filter(|b| b.strategy == BlockStrategy::Neuron)
            .collect();
        assert_eq!(neuron_blocks.len(), 2);
        for block in &neuron_blocks {
            let focus = block.focus.expect("neuron block has a focus");
            assert_ne!(creature.neurons[focus].neuron_type, "output");
        }
        // No two blocks select the same genes.
        let mut seen = HashSet::new();
        for block in blocks {
            assert!(
                seen.insert((block.neurons.clone(), block.synapses.clone())),
                "duplicate block {:?}",
                block.strategy
            );
        }
        // On this 3-neuron chain a radius-1 neighbourhood is the whole
        // creature, so both neighbourhood blocks duplicate the global one —
        // and the planner has to say how many it dropped rather than quietly
        // returning fewer blocks than were asked for.
        assert!(
            planned.dropped_duplicate >= 2,
            "expected both neighbourhood blocks to be reported as duplicates, got {}",
            planned.dropped_duplicate
        );
        assert_eq!(planned.dropped_empty, 0);
        assert!(
            !blocks
                .iter()
                .any(|b| b.strategy == BlockStrategy::Neighbourhood)
        );
    }

    /// `--radius 0` would make every neighbourhood block identical to the
    /// neuron block for the same focus, so the strategy would vanish into the
    /// duplicate filter and read as "found nothing" (#105).
    #[test]
    fn radius_zero_is_refused_for_the_neighbourhood_strategy() {
        let plan = BlockPlan {
            strategies: vec![BlockStrategy::Neuron, BlockStrategy::Neighbourhood],
            radius: 0,
            ..BlockPlan::default()
        };
        let err = plan.validate().unwrap_err();
        assert!(err.contains("radius"), "{err}");
        // A plan that never asks for a neighbourhood is unaffected.
        BlockPlan {
            strategies: vec![BlockStrategy::Neuron],
            radius: 0,
            ..BlockPlan::default()
        }
        .validate()
        .unwrap();
    }

    /// A self-loop and parallel edges must not double-weight a neuron's genes
    /// in the focus ranking or repeat a neighbour in the walk.
    #[test]
    fn adjacency_holds_each_synapse_and_neighbour_once() {
        let creature = chain();
        let graph = BlockGraph::of(&creature);
        for (i, incident) in graph.incident.iter().enumerate() {
            let mut unique = incident.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), incident.len(), "neuron {i} incident synapses");
        }
        for (i, neighbours) in graph.neighbours.iter().enumerate() {
            let mut unique = neighbours.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), neighbours.len(), "neuron {i} neighbours");
            assert!(!neighbours.contains(&i), "neuron {i} is its own neighbour");
        }
    }

    #[test]
    fn a_plan_that_could_only_produce_nothing_is_refused() {
        let creature = chain();
        let graph = BlockGraph::of(&creature);
        let magnitudes = ProposalMagnitudes::default();
        let signal = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
        let traces = vec![NeuronTraceStats::default(); creature.neurons.len()];
        let mut rng = StdRng::seed_from_u64(1);
        for plan in [
            BlockPlan {
                strategies: Vec::new(),
                ..BlockPlan::default()
            },
            BlockPlan {
                targets: TargetPlan {
                    random_control_fraction: 1.5,
                    ..TargetPlan::default()
                },
                ..BlockPlan::default()
            },
            BlockPlan {
                blocks_per_strategy: 0,
                ..BlockPlan::default()
            },
            BlockPlan {
                subgraph_size: 0,
                ..BlockPlan::default()
            },
            BlockPlan {
                top_genes: 0,
                ..BlockPlan::default()
            },
        ] {
            assert!(
                plan_blocks(
                    &creature,
                    &graph,
                    &magnitudes,
                    &signal,
                    &traces,
                    &plan,
                    &mut rng
                )
                .is_err(),
                "{plan:?} must be refused"
            );
        }
    }

    #[test]
    fn strategies_are_camel_case_on_the_wire() {
        assert_eq!(
            serde_json::to_string(&BlockStrategy::OutputHead).unwrap(),
            r#""outputHead""#
        );
        assert_eq!(BlockStrategy::TopGenes.slug(), "top-genes");
    }
}
