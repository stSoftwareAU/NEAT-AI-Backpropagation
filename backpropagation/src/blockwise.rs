//! Blockwise candidate generation: accumulate once, apply one region at a time
//! (issue #105).
//!
//! `sweep` varies *how far* the whole creature moves; this varies *what moves*.
//! One expensive accumulation pass over the corpus feeds every block
//! ([`crate::blocks`]), so a run producing twenty candidates still reads the
//! training data once for the learning signal. Each candidate is written as a
//! standalone creature and, when a scorer is supplied, scored on its own — the
//! scorer decides which region, if any, was worth moving.

use crate::backprop::{
    ApplyOptions, BackpropConfig, apply_learnings_with, calculate_learning_rate, count_apply_deltas,
};
use crate::blocks::{
    BlockGraph, BlockPlan, BlockStrategy, GeneBlock, plan_blocks, proposal_magnitudes,
};
use crate::creature_io::{ObservationWidth, load_forward_only_creature};
use crate::mse::compute_mse;
use crate::propagate_layout::accumulate_creature_learning_report;
use crate::scorer::score_creature;
use crate::targets::{
    ArmSample, SelectionComparison, TargetSelection, TargetStrategy, compare_arms,
};
use crate::train::{AcceptanceMode, resolve_acceptance};
use crate::validate::TrainedTopology;
use neat_core::{CreatureExport, compile_creature};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;

/// One selected synapse, recorded in the candidate metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlockSynapseRef {
    /// Export `synapses` position — the unambiguous identity, since a creature
    /// may hold more than one synapse between the same pair of neurons.
    pub index: usize,
    /// Source neuron UUID (`input-N` for a virtual input).
    pub from_uuid: String,
    /// Target neuron UUID.
    pub to_uuid: String,
}

/// One generated block candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlockCandidateRecord {
    /// Which strategy selected the block.
    pub strategy: BlockStrategy,
    /// Unique label, also the candidate file stem.
    pub label: String,
    /// UUID of the neuron the block was grown from, when it has one.
    #[serde(default)]
    pub focus: Option<String>,
    /// Why that target was selected and the rank features behind it — absent
    /// for the strategies that select a region rather than a target
    /// (issue #108).
    #[serde(default)]
    pub selection: Option<TargetSelection>,
    /// UUIDs of every neuron whose bias the block may move.
    pub neurons: Vec<String>,
    /// Every synapse whose weight the block may move.
    pub synapses: Vec<BlockSynapseRef>,
    /// Genes the block selected — `neurons.len() + synapses.len()`, so a block
    /// that grew large around a hub neuron is visible without counting the
    /// arrays.
    pub gene_count: usize,
    /// Hidden / constant biases that actually moved.
    pub hidden_biases: usize,
    /// Output biases that actually moved.
    pub output_biases: usize,
    /// Non-output-target synapses that actually moved.
    pub hidden_weights: usize,
    /// Output-target synapses that actually moved.
    pub output_weights: usize,
    /// Train-slice MSE of the candidate (absent under `--skip-mse`, or when no
    /// gene moved).
    #[serde(default)]
    pub train_mse: Option<f64>,
    /// `train_mse − baseline_train_mse` (negative is better).
    #[serde(default)]
    pub mse_delta: Option<f64>,
    /// Scorer fitness of the candidate (absent without a scorer).
    #[serde(default)]
    pub score: Option<f64>,
    /// `score − baseline_score` (positive is better).
    #[serde(default)]
    pub score_delta: Option<f64>,
    /// Whether the scorer gain cleared the win margin.
    #[serde(default)]
    pub score_win: Option<bool>,
    /// Wall-clock seconds this candidate's scorer run took — the cost side of
    /// the wins/hour comparison (issue #108).
    #[serde(default)]
    pub scorer_seconds: Option<f64>,
    /// Relative path of the written candidate — absent when no gene moved, so
    /// there was no candidate to write.
    #[serde(default)]
    pub candidate: Option<String>,
}

impl BlockCandidateRecord {
    /// Whether any gene of this block actually moved.
    pub fn moved(&self) -> bool {
        self.hidden_biases + self.output_biases + self.hidden_weights + self.output_weights > 0
    }
}

/// Outcome of [`run_blocks`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlocksSummary {
    /// Crate version.
    pub version: String,
    /// Records the single accumulation pass consumed.
    pub records: u64,
    /// Learning rate the proposals were built at.
    pub learning_rate: f64,
    /// Step scale every candidate applied.
    pub step_scale: f64,
    /// The plan that generated the blocks.
    pub plan: BlockPlan,
    /// Baseline train-slice MSE (absent under `--skip-mse`).
    #[serde(default)]
    pub baseline_train_mse: Option<f64>,
    /// Baseline scorer fitness (absent without a scorer).
    #[serde(default)]
    pub baseline_score: Option<f64>,
    /// Blocks whose genes all held still, so no candidate was written.
    pub unmoved_blocks: usize,
    /// Planned blocks discarded because they selected no gene.
    pub dropped_empty_blocks: usize,
    /// Planned blocks discarded because an earlier block selected the same
    /// genes — `--radius 1` around adjacent focus neurons does this.
    pub dropped_duplicate_blocks: usize,
    /// One record per planned block, in plan order. A block whose genes all
    /// held still is recorded here with `candidate: None` — use
    /// [`BlocksSummary::written`] for the candidates that reached disk.
    pub candidates: Vec<BlockCandidateRecord>,
    /// Scorer wins/hour and score gain/hour of the evidence arm beside the
    /// uniform random control arm (issue #108). Absent without a scorer —
    /// there is no throughput to compare when nothing was scored.
    #[serde(default)]
    pub selection_comparison: Option<SelectionComparison>,
}

impl BlocksSummary {
    /// Candidates that were actually written and measured.
    pub fn written(&self) -> Vec<&BlockCandidateRecord> {
        self.candidates
            .iter()
            .filter(|c| c.candidate.is_some())
            .collect()
    }

    /// Candidates whose scorer gain cleared the win margin, best first.
    pub fn winners(&self) -> Vec<&BlockCandidateRecord> {
        let mut wins: Vec<&BlockCandidateRecord> = self
            .candidates
            .iter()
            .filter(|c| c.score_win == Some(true))
            .collect();
        wins.sort_by(|a, b| {
            b.score_delta
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&a.score_delta.unwrap_or(f64::NEG_INFINITY))
        });
        wins
    }
}

/// Arguments for [`run_blocks`].
pub struct BlocksRequest<'a> {
    /// Creature JSON path (UUID-only export).
    pub creature: &'a Path,
    /// Training-data directory.
    pub training_data: &'a Path,
    /// Backprop config.
    pub config: &'a BackpropConfig,
    /// Optional record cap for the accumulation pass and the MSE checks.
    pub max_records: Option<u64>,
    /// Seed for sparse selection and the random-subgraph walks.
    pub seed: u64,
    /// Step scale every candidate applies.
    pub step_scale: f64,
    /// Which blocks to generate.
    pub plan: &'a BlockPlan,
    /// Skip every MSE pass (write and score candidates only).
    pub skip_mse: bool,
    /// Optional `rust_scorer` binary — scores the baseline and each candidate.
    pub scorer: Option<&'a Path>,
    /// Minimum scorer gain that counts as a win. Refused without a scorer.
    pub min_score_improvement: f64,
    /// Output directory for `blocks.json` and `candidates/`.
    pub output_dir: &'a Path,
}

/// File-system-safe stem for a candidate label.
fn slug(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Describe a block's selected genes by UUID, for the candidate metadata.
fn describe(creature: &CreatureExport, block: &GeneBlock) -> (Vec<String>, Vec<BlockSynapseRef>) {
    let neurons = block
        .neurons
        .iter()
        .filter_map(|&i| creature.neurons.get(i).map(|n| n.uuid.clone()))
        .collect();
    let synapses = block
        .synapses
        .iter()
        .filter_map(|&i| {
            creature.synapses.get(i).map(|s| BlockSynapseRef {
                index: i,
                from_uuid: s.from_uuid.clone(),
                to_uuid: s.to_uuid.clone(),
            })
        })
        .collect();
    (neurons, synapses)
}

/// Accumulate once, then write one candidate per block (issue #105).
pub fn run_blocks(req: BlocksRequest<'_>) -> Result<BlocksSummary, String> {
    req.plan.validate()?;
    // Reuse the acceptance rule: a win margin on a run with no scorer is a
    // misconfiguration, not a no-op — the run would look gated and score
    // nothing.
    let margin = match resolve_acceptance(req.scorer.is_some(), req.min_score_improvement, false)? {
        AcceptanceMode::Scorer(settings) => Some(settings.min_improvement),
        AcceptanceMode::Mse => None,
    };
    // A sparse pass only accumulates for the neurons its own random draw
    // selected, so every evidence term is zero for the rest and the ranking
    // degenerates to "rank that random subset". Said out loud rather than left
    // to be discovered in the rank features.
    if req.plan.targets.strategy == TargetStrategy::Evidence && req.config.sparse_ratio < 1.0 {
        eprintln!(
            "blocks: warning — sparse_ratio {} means only that random share of neurons \
             accumulated any signal, so evidence target selection ranks that subset alone",
            req.config.sparse_ratio
        );
    }
    let incumbent = load_forward_only_creature(req.creature)?;
    let width = ObservationWidth::of(&incumbent)?;
    let topology = TrainedTopology::of(&incumbent);
    let candidates_dir = req.output_dir.join("candidates");
    fs::create_dir_all(&candidates_dir).map_err(|e| e.to_string())?;

    let learning_rate = calculate_learning_rate(req.config, 0, None);
    let baseline_train_mse = if req.skip_mse {
        None
    } else {
        let mut network = compile_creature(&incumbent).map_err(|e| e.to_string())?;
        Some(compute_mse(&incumbent, &mut network, req.training_data, req.max_records)?.0)
    };
    let score_dir = req.output_dir.join("scorer-work");
    let baseline_score = match req.scorer {
        Some(scorer) => Some(
            score_creature(
                scorer,
                &width.checked_json_pretty(&incumbent)?,
                req.training_data,
                &score_dir.join("baseline"),
            )?
            .score,
        ),
        None => None,
    };

    // The one expensive pass. Every block below is carved out of this signal,
    // so the corpus is read once for the learning itself.
    let mut rng = StdRng::seed_from_u64(req.seed);
    let mut network = compile_creature(&incumbent).map_err(|e| e.to_string())?;
    let report = accumulate_creature_learning_report(
        &incumbent,
        &mut network,
        req.training_data,
        req.config,
        req.max_records,
        &mut rng,
    )?;

    let graph = BlockGraph::of(&incumbent);
    let magnitudes = proposal_magnitudes(
        &incumbent,
        &report.learning,
        req.config,
        learning_rate,
        req.step_scale,
    );
    let planned = plan_blocks(
        &incumbent,
        &graph,
        &magnitudes,
        &report.learning,
        &report.neuron_traces,
        req.plan,
        &mut rng,
    )?;
    if planned.dropped_empty + planned.dropped_duplicate > 0 {
        eprintln!(
            "blocks: dropped {} empty and {} duplicate block(s) from the plan",
            planned.dropped_empty, planned.dropped_duplicate
        );
    }

    let apply = ApplyOptions {
        step_scale: req.step_scale,
        outputs_only: false,
        hidden_only: false,
    };
    let mut records = Vec::with_capacity(planned.blocks.len());
    let mut unmoved_blocks = 0usize;
    let mut arm_samples: Vec<ArmSample> = Vec::new();
    for (index, block) in planned.blocks.iter().enumerate() {
        let focus_uuid = block
            .focus
            .and_then(|i| incumbent.neurons.get(i))
            .map(|n| n.uuid.clone());
        let label = match &focus_uuid {
            Some(uuid) => format!("{index:03}-{}-{}", block.strategy.slug(), slug(uuid)),
            None => format!("{index:03}-{}", block.strategy.slug()),
        };
        let candidate = apply_learnings_with(
            &incumbent,
            &block.mask(&report.learning),
            req.config,
            learning_rate,
            apply,
        );
        let deltas = count_apply_deltas(&incumbent, &candidate, req.config.plank_constant);
        let (neurons, synapses) = describe(&incumbent, block);
        let mut record = BlockCandidateRecord {
            strategy: block.strategy,
            label: label.clone(),
            focus: focus_uuid,
            selection: block.selection.clone(),
            neurons,
            synapses,
            gene_count: block.gene_count(),
            hidden_biases: deltas.hidden_biases,
            output_biases: deltas.output_biases,
            hidden_weights: deltas.hidden_weights,
            output_weights: deltas.output_weights,
            train_mse: None,
            mse_delta: None,
            score: None,
            score_delta: None,
            score_win: None,
            scorer_seconds: None,
            candidate: None,
        };
        if !record.moved() {
            // Nothing moved, so the "candidate" is the incumbent: writing and
            // scoring it would spend a full scorer run to learn the baseline
            // again. Counted in the summary rather than dropped silently.
            unmoved_blocks += 1;
            eprintln!("blocks {label}: no gene moved — skipped");
            records.push(record);
            continue;
        }
        // Issue #94: a block candidate is a trained creature that reaches disk,
        // so it is gated the same way — once, as it is produced.
        topology.assert_valid(&candidate, &format!("block candidate {label}"))?;
        let json = width.checked_json_pretty(&candidate)?;
        let file = format!("{label}.json");
        fs::write(candidates_dir.join(&file), &json).map_err(|e| e.to_string())?;
        record.candidate = Some(format!("candidates/{file}"));

        if let Some(baseline) = baseline_train_mse {
            let mut candidate_net = compile_creature(&candidate).map_err(|e| e.to_string())?;
            let mse = compute_mse(
                &candidate,
                &mut candidate_net,
                req.training_data,
                req.max_records,
            )?
            .0;
            record.train_mse = Some(mse);
            record.mse_delta = Some(mse - baseline);
        }
        if let (Some(scorer), Some(baseline), Some(margin)) = (req.scorer, baseline_score, margin) {
            // The scorer run is the scarce resource issue #108 is about, so
            // its wall clock is measured rather than estimated.
            let started = Instant::now();
            let scored = score_creature(
                scorer,
                &json,
                req.training_data,
                &score_dir.join("candidate"),
            )?;
            let seconds = started.elapsed().as_secs_f64();
            let delta = scored.score - baseline;
            let win = delta >= margin;
            record.score = Some(scored.score);
            record.score_delta = Some(delta);
            record.score_win = Some(win);
            record.scorer_seconds = Some(seconds);
            if let Some(selection) = &block.selection {
                arm_samples.push(ArmSample {
                    source: selection.source,
                    score_delta: Some(delta),
                    win,
                    scorer_seconds: seconds,
                });
            }
        }
        eprintln!(
            "blocks {label}: genes={} moved={} mse_delta={} score_delta={}",
            block.gene_count(),
            deltas.hidden_biases
                + deltas.output_biases
                + deltas.hidden_weights
                + deltas.output_weights,
            record
                .mse_delta
                .map_or_else(|| "-".into(), |v| format!("{v:+.6e}")),
            record
                .score_delta
                .map_or_else(|| "-".into(), |v| format!("{v:+.6e}")),
        );
        records.push(record);
    }

    let summary = BlocksSummary {
        version: env!("CARGO_PKG_VERSION").to_string(),
        records: report.records,
        learning_rate,
        step_scale: req.step_scale,
        plan: req.plan.clone(),
        baseline_train_mse,
        baseline_score,
        unmoved_blocks,
        dropped_empty_blocks: planned.dropped_empty,
        dropped_duplicate_blocks: planned.dropped_duplicate,
        candidates: records,
        selection_comparison: (!arm_samples.is_empty()).then(|| compare_arms(&arm_samples)),
    };
    fs::write(
        req.output_dir.join("blocks.json"),
        serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::DEFAULT_STEP_SCALE;
    use std::io::Write;
    use tempfile::tempdir;

    /// `input-0 → h1 → h2 → o1` with a `h1 → o1` shortcut, so blocks around h1
    /// and around the output head select genuinely different genes.
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
        {"fromUUID":"h1","toUUID":"o1","weight":0.5},
        {"fromUUID":"h2","toUUID":"o1","weight":0.5}
      ]
    }"#;

    /// Creature + corpus in a temp dir; returns (dir, creature path, data dir).
    fn fixture() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        let dir = tempdir().unwrap();
        let data = dir.path().join("data");
        fs::create_dir_all(&data).unwrap();
        let mut f = fs::File::create(data.join("0.bin")).unwrap();
        for i in 0..16u32 {
            let x = i as f32 / 8.0 - 1.0;
            f.write_all(&x.to_le_bytes()).unwrap();
            f.write_all(&(0.5 * x + 0.25).to_le_bytes()).unwrap();
        }
        let creature = dir.path().join("creature.json");
        fs::write(&creature, CHAIN).unwrap();
        (dir, creature, data)
    }

    fn request<'a>(
        creature: &'a Path,
        data: &'a Path,
        config: &'a BackpropConfig,
        plan: &'a BlockPlan,
        out: &'a Path,
    ) -> BlocksRequest<'a> {
        BlocksRequest {
            creature,
            training_data: data,
            config,
            max_records: None,
            seed: 1,
            step_scale: DEFAULT_STEP_SCALE,
            plan,
            skip_mse: false,
            scorer: None,
            min_score_improvement: crate::train::DEFAULT_MIN_SCORE_IMPROVEMENT,
            output_dir: out,
        }
    }

    #[test]
    fn writes_one_candidate_per_block_with_its_metadata() {
        let (dir, creature, data) = fixture();
        let out = dir.path().join("out");
        let config = BackpropConfig::default();
        let plan = BlockPlan {
            blocks_per_strategy: 2,
            subgraph_size: 2,
            top_genes: 3,
            ..BlockPlan::default()
        };
        let summary = run_blocks(request(&creature, &data, &config, &plan, &out)).unwrap();

        assert!(summary.candidates.len() > 1, "expected several blocks");
        assert!(summary.records > 0);
        assert!(out.join("blocks.json").is_file());
        for candidate in &summary.candidates {
            if !candidate.moved() {
                assert!(candidate.candidate.is_none());
                continue;
            }
            let path = out.join(candidate.candidate.as_ref().unwrap());
            assert!(path.is_file(), "{} written", path.display());
            // Metadata names the genes the block selected.
            assert!(
                candidate.neurons.len() + candidate.synapses.len() > 0,
                "{} selects genes",
                candidate.label
            );
            assert!(candidate.train_mse.is_some());
            assert!(candidate.score.is_none(), "no scorer was supplied");
        }
        // The whole-creature apply is still available for parity.
        let global = summary
            .candidates
            .iter()
            .find(|c| c.strategy == BlockStrategy::Global)
            .expect("global block present");
        assert_eq!(global.neurons.len(), 3);
        assert_eq!(global.synapses.len(), 4);
    }

    /// The point of the mode: a block candidate must move only its own genes,
    /// while the global candidate moves the lot.
    #[test]
    fn a_block_candidate_moves_only_the_genes_it_names() {
        let (dir, creature, data) = fixture();
        let out = dir.path().join("out");
        let config = BackpropConfig::default();
        let plan = BlockPlan {
            strategies: vec![BlockStrategy::Global, BlockStrategy::Neuron],
            blocks_per_strategy: 1,
            ..BlockPlan::default()
        };
        let summary = run_blocks(request(&creature, &data, &config, &plan, &out)).unwrap();
        let source = load_forward_only_creature(&creature).unwrap();

        let neuron_block = summary
            .candidates
            .iter()
            .find(|c| c.strategy == BlockStrategy::Neuron)
            .expect("neuron block present");
        let applied = load_forward_only_creature(
            &out.join(neuron_block.candidate.as_ref().expect("candidate written")),
        )
        .unwrap();
        for (i, neuron) in applied.neurons.iter().enumerate() {
            if !neuron_block.neurons.contains(&neuron.uuid) {
                assert_eq!(
                    neuron.bias, source.neurons[i].bias,
                    "{} is outside the block and must hold",
                    neuron.uuid
                );
            }
        }
        let moved_weights = applied
            .synapses
            .iter()
            .zip(source.synapses.iter())
            .filter(|(a, b)| (a.weight - b.weight).abs() >= config.plank_constant)
            .count();
        assert!(moved_weights > 0, "the block moved something");
        assert!(
            moved_weights <= neuron_block.synapses.len(),
            "moved {moved_weights} weights but named {}",
            neuron_block.synapses.len()
        );
        // The global candidate is the whole-creature apply, so it moves more.
        let global = summary
            .candidates
            .iter()
            .find(|c| c.strategy == BlockStrategy::Global)
            .expect("global block present");
        assert!(
            global.hidden_biases
                + global.output_biases
                + global.hidden_weights
                + global.output_weights
                > neuron_block.hidden_biases
                    + neuron_block.output_biases
                    + neuron_block.hidden_weights
                    + neuron_block.output_weights,
            "global must move more genes than one neuron block"
        );
    }

    #[test]
    fn a_win_margin_without_a_scorer_is_refused() {
        let (dir, creature, data) = fixture();
        let out = dir.path().join("out");
        let config = BackpropConfig::default();
        let plan = BlockPlan::default();
        let mut req = request(&creature, &data, &config, &plan, &out);
        req.min_score_improvement = 0.5;
        let err = run_blocks(req).unwrap_err();
        assert!(err.contains("minScoreImprovement"), "{err}");
    }

    #[test]
    fn skip_mse_writes_candidates_without_measuring_them() {
        let (dir, creature, data) = fixture();
        let out = dir.path().join("out");
        let config = BackpropConfig::default();
        let plan = BlockPlan {
            strategies: vec![BlockStrategy::OutputHead],
            ..BlockPlan::default()
        };
        let mut req = request(&creature, &data, &config, &plan, &out);
        req.skip_mse = true;
        let summary = run_blocks(req).unwrap();
        assert!(summary.baseline_train_mse.is_none());
        assert_eq!(summary.candidates.len(), 1);
        assert!(summary.candidates[0].train_mse.is_none());
        assert!(summary.candidates[0].candidate.is_some());
    }
}
