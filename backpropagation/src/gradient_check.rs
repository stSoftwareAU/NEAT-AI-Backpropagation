//! Finite-difference probe: proposal Δ vs numerical ∂MSE/∂gene (issue #40).
//!
//! Accumulates learning once, samples genes with a non-zero proposal, and
//! compares each proposal delta against a central finite-difference estimate
//! of ∂MSE/∂θ on the same record slice.
//!
//! For MSE minimization a descent proposal satisfies
//! `proposal_delta · ∂MSE/∂θ < 0`. Sign agreement is that predicate (genes
//! with near-zero FD are excluded from the percentage).

use crate::backprop::{BackpropConfig, LearningSignal, calculate_learning_rate};
use crate::mse::compute_mse;
use crate::propagate_layout::accumulate_creature_learning_report;
use neat_core::{CreatureExport, compile_creature, parse_creature_json};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Gene class used for stratified reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GeneClass {
    /// Hidden / constant neuron bias.
    HiddenBias,
    /// Output neuron bias.
    OutputBias,
    /// Synapse that does not target an output.
    HiddenWeight,
    /// Synapse that targets an output.
    OutputWeight,
}

impl GeneClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::HiddenBias => "hiddenBias",
            Self::OutputBias => "outputBias",
            Self::HiddenWeight => "hiddenWeight",
            Self::OutputWeight => "outputWeight",
        }
    }
}

/// One sampled gene in the probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeneProbeRow {
    /// Gene class.
    pub class: GeneClass,
    /// Export neuron or synapse index.
    pub index: usize,
    /// UUID (bias) or `fromUUID→toUUID` (weight).
    pub id: String,
    /// Current gene value.
    pub current: f64,
    /// Proposal delta after step scale (proposed − current)×step.
    pub proposal_delta: f64,
    /// Central finite-difference ∂MSE/∂θ.
    pub fd_grad: f64,
    /// True when `proposal_delta · fd_grad < 0` (descent agreement).
    pub sign_agree: bool,
    /// `|proposal_delta| / |fd_grad|` when `|fd_grad|` is above the FD floor.
    pub magnitude_ratio: Option<f64>,
}

/// Aggregate stats for one gene class.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ClassStats {
    /// Gene class label.
    pub class: String,
    /// Genes sampled in this class.
    pub sampled: usize,
    /// Genes with `|fd_grad|` above the floor (included in sign %).
    pub scored: usize,
    /// Count where proposal and −FD share a descent direction.
    pub sign_agree: usize,
    /// `sign_agree / scored` (0 when scored is 0).
    pub sign_agree_pct: f64,
    /// Median `|proposal|/|fd|` over scored genes with a ratio.
    pub magnitude_ratio_p50: Option<f64>,
    /// 90th percentile of that ratio.
    pub magnitude_ratio_p90: Option<f64>,
}

/// Outcome of [`run_gradient_check`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GradientCheckSummary {
    /// Crate version.
    pub version: String,
    /// Issue this probe addresses.
    pub issue: u32,
    /// Baseline MSE on the probe slice.
    pub baseline_mse: f64,
    /// Records used for accumulate and FD MSE.
    pub records: u64,
    /// Finite-difference ε.
    pub fd_eps: f64,
    /// Step scale applied to proposals.
    pub step_scale: f64,
    /// Learning rate used for proposals.
    pub learning_rate: f64,
    /// Per-class aggregates.
    pub by_class: Vec<ClassStats>,
    /// Overall scored genes.
    pub scored: usize,
    /// Overall sign agreements.
    pub sign_agree: usize,
    /// Overall sign agreement percentage.
    pub sign_agree_pct: f64,
    /// Per-gene rows (same order as written to `genes.jsonl`).
    pub genes: Vec<GeneProbeRow>,
}

/// Arguments for [`run_gradient_check`].
pub struct GradientCheckRequest<'a> {
    /// Creature JSON path.
    pub creature: &'a Path,
    /// Training-data directory.
    pub training_data: &'a Path,
    /// Backprop config.
    pub config: &'a BackpropConfig,
    /// Optional record cap (accumulate + FD MSE).
    pub max_records: Option<u64>,
    /// RNG seed (accumulate sparse selection + gene sampling).
    pub seed: u64,
    /// Max bias genes to sample (stratified across classes).
    pub sample_biases: usize,
    /// Max weight genes to sample (stratified across classes).
    pub sample_weights: usize,
    /// Central finite-difference ε.
    pub fd_eps: f64,
    /// Step scale applied to (proposed − current).
    pub step_scale: f64,
    /// Restrict eligible pool to output genes.
    pub outputs_only: bool,
    /// Restrict eligible pool to hidden genes.
    pub hidden_only: bool,
    /// Output directory for `gradient-check.json` + `genes.jsonl`.
    pub output_dir: &'a Path,
}

/// Accumulate once and compare proposal deltas to finite-difference gradients.
pub fn run_gradient_check(req: GradientCheckRequest<'_>) -> Result<GradientCheckSummary, String> {
    if req.outputs_only && req.hidden_only {
        return Err("gradient-check cannot set both outputs-only and hidden-only".into());
    }
    if !(req.fd_eps.is_finite() && req.fd_eps > 0.0) {
        return Err("fd-eps must be positive and finite".into());
    }
    fs::create_dir_all(req.output_dir).map_err(|e| e.to_string())?;
    let text = fs::read_to_string(req.creature).map_err(|e| e.to_string())?;
    let creature = parse_creature_json(&text).map_err(|e| e.to_string())?;
    if !creature.forward_only {
        return Err(
            "this trainer supports forward-only creatures only (no re-entrant / recurrent graphs)"
                .into(),
        );
    }

    let lr = calculate_learning_rate(req.config, 0, None);
    let step = if req.step_scale.is_finite() && req.step_scale > 0.0 {
        req.step_scale.min(1.0)
    } else {
        1.0
    };

    let mut network = compile_creature(&creature).map_err(|e| e.to_string())?;
    let (baseline_mse, _) =
        compute_mse(&creature, &mut network, req.training_data, req.max_records)?;

    let mut rng = StdRng::seed_from_u64(req.seed);
    let mut net = compile_creature(&creature).map_err(|e| e.to_string())?;
    let report = accumulate_creature_learning_report(
        &creature,
        &mut net,
        req.training_data,
        req.config,
        req.max_records,
        &mut rng,
    )?;

    let output_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    let filter = PoolFilter {
        outputs_only: req.outputs_only,
        hidden_only: req.hidden_only,
        lr,
        step,
        output_uuids: &output_uuids,
    };
    let bias_pool = eligible_biases(&creature, &report.learning, req.config, &filter);
    let weight_pool = eligible_weights(&creature, &report.learning, req.config, &filter);

    let sampled_biases = stratified_sample(&bias_pool, req.sample_biases, &mut rng);
    let sampled_weights = stratified_sample(&weight_pool, req.sample_weights, &mut rng);

    let fd = FdCtx {
        eps: req.fd_eps,
        fd_floor: req.config.plank_constant.max(1e-12),
        training_data: req.training_data,
        max_records: req.max_records,
    };
    let mut genes = Vec::with_capacity(sampled_biases.len() + sampled_weights.len());

    for gene in &sampled_biases {
        genes.push(probe_bias_gene(&creature, gene, &fd)?);
    }
    for gene in &sampled_weights {
        genes.push(probe_weight_gene(&creature, gene, &fd)?);
    }

    let by_class = aggregate_by_class(&genes);
    let scored: usize = by_class.iter().map(|c| c.scored).sum();
    let sign_agree: usize = by_class.iter().map(|c| c.sign_agree).sum();
    let sign_agree_pct = if scored == 0 {
        0.0
    } else {
        100.0 * sign_agree as f64 / scored as f64
    };

    let summary = GradientCheckSummary {
        version: env!("CARGO_PKG_VERSION").to_string(),
        issue: 40,
        baseline_mse,
        records: report.records,
        fd_eps: req.fd_eps,
        step_scale: step,
        learning_rate: lr,
        by_class,
        scored,
        sign_agree,
        sign_agree_pct,
        genes: genes.clone(),
    };

    let mut genes_jsonl = String::new();
    for row in &genes {
        genes_jsonl.push_str(&serde_json::to_string(row).map_err(|e| e.to_string())?);
        genes_jsonl.push('\n');
    }
    fs::write(req.output_dir.join("genes.jsonl"), genes_jsonl).map_err(|e| e.to_string())?;
    // Summary without the bulky gene list for the pretty JSON (genes live in jsonl).
    let mut summary_file = summary.clone();
    summary_file.genes.clear();
    fs::write(
        req.output_dir.join("gradient-check.json"),
        serde_json::to_string_pretty(&summary_file).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    Ok(summary)
}

#[derive(Debug, Clone)]
struct EligibleGene {
    class: GeneClass,
    index: usize,
    current: f64,
    proposal_delta: f64,
    id: String,
}

struct PoolFilter<'a> {
    outputs_only: bool,
    hidden_only: bool,
    lr: f64,
    step: f64,
    output_uuids: &'a HashSet<&'a str>,
}

struct FdCtx<'a> {
    eps: f64,
    fd_floor: f64,
    training_data: &'a Path,
    max_records: Option<u64>,
}

fn eligible_biases(
    creature: &CreatureExport,
    signal: &LearningSignal,
    config: &BackpropConfig,
    filter: &PoolFilter<'_>,
) -> Vec<EligibleGene> {
    let mut out = Vec::new();
    for (i, neuron) in creature.neurons.iter().enumerate() {
        let is_output = neuron.neuron_type == "output";
        if filter.outputs_only && !is_output {
            continue;
        }
        if filter.hidden_only && is_output {
            continue;
        }
        let Some(sig) = signal.biases.get(i) else {
            continue;
        };
        if sig.count <= 0.0 {
            continue;
        }
        let proposed = sig.propose(neuron.bias, config, filter.lr);
        let delta = (proposed - neuron.bias) * filter.step;
        if delta.abs() < config.plank_constant {
            continue;
        }
        let class = if is_output {
            GeneClass::OutputBias
        } else {
            GeneClass::HiddenBias
        };
        out.push(EligibleGene {
            class,
            index: i,
            current: neuron.bias,
            proposal_delta: delta,
            id: neuron.uuid.clone(),
        });
    }
    out
}

fn eligible_weights(
    creature: &CreatureExport,
    signal: &LearningSignal,
    config: &BackpropConfig,
    filter: &PoolFilter<'_>,
) -> Vec<EligibleGene> {
    let mut out = Vec::new();
    for (i, syn) in creature.synapses.iter().enumerate() {
        let targets_output = filter.output_uuids.contains(syn.to_uuid.as_str());
        if filter.outputs_only && !targets_output {
            continue;
        }
        if filter.hidden_only && targets_output {
            continue;
        }
        let Some(sig) = signal.weights.get(i) else {
            continue;
        };
        if sig.count <= 0.0 {
            continue;
        }
        let proposed = sig.propose(syn.weight, config, filter.lr);
        let delta = (proposed - syn.weight) * filter.step;
        if delta.abs() < config.plank_constant {
            continue;
        }
        let class = if targets_output {
            GeneClass::OutputWeight
        } else {
            GeneClass::HiddenWeight
        };
        out.push(EligibleGene {
            class,
            index: i,
            current: syn.weight,
            proposal_delta: delta,
            id: format!("{}→{}", syn.from_uuid, syn.to_uuid),
        });
    }
    out
}

fn stratified_sample(pool: &[EligibleGene], budget: usize, rng: &mut StdRng) -> Vec<EligibleGene> {
    if budget == 0 || pool.is_empty() {
        return Vec::new();
    }
    let mut by_class: Vec<(GeneClass, Vec<&EligibleGene>)> = Vec::new();
    for row in pool {
        if let Some((_, bucket)) = by_class.iter_mut().find(|(c, _)| *c == row.class) {
            bucket.push(row);
        } else {
            by_class.push((row.class, vec![row]));
        }
    }
    for (_, bucket) in &mut by_class {
        bucket.shuffle(rng);
    }
    let mut selected: Vec<EligibleGene> = Vec::new();
    // Round-robin across classes so each gets coverage.
    let mut idx = vec![0usize; by_class.len()];
    while selected.len() < budget {
        let mut progressed = false;
        for (ci, (_, bucket)) in by_class.iter().enumerate() {
            if selected.len() >= budget {
                break;
            }
            if idx[ci] < bucket.len() {
                selected.push(bucket[idx[ci]].clone());
                idx[ci] += 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    selected
}

fn probe_bias_gene(
    creature: &CreatureExport,
    gene: &EligibleGene,
    fd: &FdCtx<'_>,
) -> Result<GeneProbeRow, String> {
    let mse_plus = mse_with_bias(
        creature,
        gene.index,
        gene.current + fd.eps,
        fd.training_data,
        fd.max_records,
    )?;
    let mse_minus = mse_with_bias(
        creature,
        gene.index,
        gene.current - fd.eps,
        fd.training_data,
        fd.max_records,
    )?;
    let fd_grad = (mse_plus - mse_minus) / (2.0 * fd.eps);
    Ok(finalize_row(gene, fd_grad, fd.fd_floor))
}

fn probe_weight_gene(
    creature: &CreatureExport,
    gene: &EligibleGene,
    fd: &FdCtx<'_>,
) -> Result<GeneProbeRow, String> {
    let mse_plus = mse_with_weight(
        creature,
        gene.index,
        gene.current + fd.eps,
        fd.training_data,
        fd.max_records,
    )?;
    let mse_minus = mse_with_weight(
        creature,
        gene.index,
        gene.current - fd.eps,
        fd.training_data,
        fd.max_records,
    )?;
    let fd_grad = (mse_plus - mse_minus) / (2.0 * fd.eps);
    Ok(finalize_row(gene, fd_grad, fd.fd_floor))
}

fn mse_with_bias(
    creature: &CreatureExport,
    index: usize,
    value: f64,
    training_data: &Path,
    max_records: Option<u64>,
) -> Result<f64, String> {
    let mut mutated = creature.clone();
    mutated.neurons[index].bias = value;
    let mut net = compile_creature(&mutated).map_err(|e| e.to_string())?;
    Ok(compute_mse(&mutated, &mut net, training_data, max_records)?.0)
}

fn mse_with_weight(
    creature: &CreatureExport,
    index: usize,
    value: f64,
    training_data: &Path,
    max_records: Option<u64>,
) -> Result<f64, String> {
    let mut mutated = creature.clone();
    mutated.synapses[index].weight = value;
    let mut net = compile_creature(&mutated).map_err(|e| e.to_string())?;
    Ok(compute_mse(&mutated, &mut net, training_data, max_records)?.0)
}

fn finalize_row(gene: &EligibleGene, fd_grad: f64, fd_floor: f64) -> GeneProbeRow {
    let scored = fd_grad.is_finite() && fd_grad.abs() >= fd_floor;
    let sign_agree = scored && gene.proposal_delta * fd_grad < 0.0;
    let magnitude_ratio = if scored {
        Some(gene.proposal_delta.abs() / fd_grad.abs())
    } else {
        None
    };
    GeneProbeRow {
        class: gene.class,
        index: gene.index,
        id: gene.id.clone(),
        current: gene.current,
        proposal_delta: gene.proposal_delta,
        fd_grad,
        sign_agree,
        magnitude_ratio,
    }
}

fn aggregate_by_class(genes: &[GeneProbeRow]) -> Vec<ClassStats> {
    let order = [
        GeneClass::OutputBias,
        GeneClass::HiddenBias,
        GeneClass::OutputWeight,
        GeneClass::HiddenWeight,
    ];
    let mut stats = Vec::new();
    for class in order {
        let rows: Vec<&GeneProbeRow> = genes.iter().filter(|g| g.class == class).collect();
        if rows.is_empty() {
            continue;
        }
        let sampled = rows.len();
        // `magnitude_ratio` is Some iff FD cleared the floor in finalize_row.
        let scored_rows: Vec<&GeneProbeRow> = rows
            .iter()
            .copied()
            .filter(|g| g.magnitude_ratio.is_some())
            .collect();
        let scored = scored_rows.len();
        let sign_agree = scored_rows.iter().filter(|g| g.sign_agree).count();
        let mut ratios: Vec<f64> = scored_rows
            .iter()
            .filter_map(|g| g.magnitude_ratio)
            .collect();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let sign_agree_pct = if scored == 0 {
            0.0
        } else {
            100.0 * sign_agree as f64 / scored as f64
        };
        stats.push(ClassStats {
            class: class.as_str().to_string(),
            sampled,
            scored,
            sign_agree,
            sign_agree_pct,
            magnitude_ratio_p50: percentile(&ratios, 0.50),
            magnitude_ratio_p90: percentile(&ratios, 0.90),
        });
    }
    stats
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    Some(sorted[idx.min(sorted.len() - 1)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backprop::BackpropConfig;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn identity_chain_output_genes_agree_with_fd() {
        let dir = tempdir().unwrap();
        let data = dir.path().join("data");
        fs::create_dir_all(&data).unwrap();
        let mut f = fs::File::create(data.join("0.bin")).unwrap();
        // input=1 → identity chain pred=1; target=2 → error to learn.
        f.write_all(&1.0f32.to_le_bytes()).unwrap();
        f.write_all(&2.0f32.to_le_bytes()).unwrap();
        let creature_path = dir.path().join("creature.json");
        fs::write(
            &creature_path,
            r#"{
              "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
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
        let out = dir.path().join("out");
        let cfg = BackpropConfig::default();
        let summary = run_gradient_check(GradientCheckRequest {
            creature: &creature_path,
            training_data: &data,
            config: &cfg,
            max_records: Some(1),
            seed: 1,
            sample_biases: 8,
            sample_weights: 8,
            fd_eps: 1e-4,
            step_scale: 1.0,
            outputs_only: false,
            hidden_only: false,
            output_dir: &out,
        })
        .unwrap();
        assert!(out.join("gradient-check.json").is_file());
        assert!(out.join("genes.jsonl").is_file());
        assert!(summary.baseline_mse > 0.0);

        let output_rows: Vec<_> = summary
            .genes
            .iter()
            .filter(|g| matches!(g.class, GeneClass::OutputBias | GeneClass::OutputWeight))
            .filter(|g| g.magnitude_ratio.is_some())
            .collect();
        assert!(
            !output_rows.is_empty(),
            "expected at least one scored output gene"
        );
        for row in &output_rows {
            assert!(
                row.sign_agree,
                "output gene {} should be a descent proposal (Δ={} fd={})",
                row.id, row.proposal_delta, row.fd_grad
            );
        }
    }
}
