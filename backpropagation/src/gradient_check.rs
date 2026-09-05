//! Finite-difference probe: proposal Δ vs numerical ∂MSE/∂gene (issue #40).
//!
//! Accumulates learning once, samples genes with a non-zero proposal, and
//! compares each proposal delta against a central finite-difference estimate
//! of ∂MSE/∂θ on the same record slice.
//!
//! For MSE minimization a descent proposal satisfies
//! `proposal_delta · ∂MSE/∂θ < 0`. Sign agreement is that predicate (genes
//! with near-zero FD are excluded from the percentage).
//!
//! Issue #107 widened the report from four gene classes to a full facet
//! stratification — squash, aggregate vs ordinary, depth, fan-in / fan-out,
//! activation health and proposal magnitude — and added the ground truth
//! behind the gradient: each sampled gene is also *applied* on its own so the
//! artefact records whether the proposal actually lowered slice MSE, and how
//! far the first-order prediction `fd_grad · Δ` was from that outcome. See
//! [`crate::gene_facets`] for the labelling and ranking.

use crate::backprop::{
    BackpropConfig, LearningSignal, calculate_learning_rate, effective_step_scale,
};
use crate::creature_io::load_forward_only_creature;
use crate::gene_facets::{
    CreatureTopology, FacetRow, FacetStats, GeneAttributes, aggregate_facets, attributes_for,
    percentage, percentile, rank_facets, sorted_values,
};
use crate::mse::compute_mse;
use crate::propagate_layout::accumulate_creature_learning_report;
use neat_core::{CreatureExport, compile_creature};
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

    /// `bias` or `weight` — the gene-kind facet.
    fn gene_kind(self) -> &'static str {
        match self {
            Self::HiddenBias | Self::OutputBias => "bias",
            Self::HiddenWeight | Self::OutputWeight => "weight",
        }
    }

    /// `output` or `hidden` — the role facet.
    fn role(self) -> &'static str {
        match self {
            Self::OutputBias | Self::OutputWeight => "output",
            Self::HiddenBias | Self::HiddenWeight => "hidden",
        }
    }

    /// True when the gene is a neuron bias rather than a synapse weight.
    fn is_bias(self) -> bool {
        matches!(self, Self::HiddenBias | Self::OutputBias)
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
    /// Squash, aggregate flag, depth, degrees and activation health of the
    /// gene's neuron (issue #107).
    pub attributes: GeneAttributes,
    /// Gradient implied by the proposal, `−proposal_delta / (lr · step)` —
    /// the descent step inverted back through the learning rate and step
    /// scale, so it is comparable with `fd_grad`.
    pub proposal_grad: f64,
    /// `|proposal_grad − fd_grad|` — the absolute gradient error.
    pub grad_abs_error: f64,
    /// `grad_abs_error / max(|proposal_grad|, |fd_grad|)` — the relative
    /// gradient error. `None` for a gene the finite difference could not
    /// score (below the FD floor, or a non-finite scale), so an unmeasured
    /// gene never lands in an error distribution as a spurious `1.0`.
    pub grad_rel_error: Option<f64>,
    /// First-order predicted MSE change, `fd_grad · proposal_delta`.
    pub predicted_delta_mse: f64,
    /// Measured MSE change from applying the proposal to this gene alone.
    pub actual_delta_mse: f64,
    /// True when applying the proposal actually lowered slice MSE.
    pub improved: bool,
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

/// Artefact schema version of `gradient-check.json` / `genes.jsonl`.
///
/// Bumped to `2` by issue #107 (facets, applied-proposal ground truth). A
/// consumer comparing artefacts across NEAT-AI-core / Backpropagation
/// versions reads this first and refuses a schema it does not know.
pub const GRADIENT_CHECK_SCHEMA: u32 = 2;

/// The repository's declared neat-core baseline, as committed in
/// `neat-core.expected-version` (issue #107).
///
/// neat-core is an unpinned `path` dependency tracking head, so the crate
/// version alone cannot tell two artefacts apart when the difference came
/// from core. Stamping the handled baseline beside it makes the comparison
/// legible: a comment-and-blank-line header followed by the version.
const NEAT_CORE_EXPECTED_VERSION: &str = include_str!("../../neat-core.expected-version");

/// The version line of [`NEAT_CORE_EXPECTED_VERSION`], or `"unknown"` when
/// the file carries no version line.
fn neat_core_baseline() -> String {
    NEAT_CORE_EXPECTED_VERSION
        .lines()
        .map(str::trim)
        .rfind(|line| !line.is_empty() && !line.starts_with('#'))
        .unwrap_or("unknown")
        .to_string()
}

/// Shape of the probed creature, so two artefacts can be shown to describe
/// the same network before their numbers are compared (issue #107).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CreatureFingerprint {
    /// Observation width — top-level `input`.
    pub input: usize,
    /// Observation width — top-level `output`.
    pub output: usize,
    /// Non-input neuron count.
    pub neurons: usize,
    /// Synapse count.
    pub synapses: usize,
}

impl CreatureFingerprint {
    /// Read the fingerprint off a creature export.
    pub fn of(creature: &CreatureExport) -> Self {
        Self {
            input: creature.input,
            output: creature.output,
            neurons: creature.neurons.len(),
            synapses: creature.synapses.len(),
        }
    }
}

/// Outcome of [`run_gradient_check`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GradientCheckSummary {
    /// Artefact schema version — [`GRADIENT_CHECK_SCHEMA`].
    pub schema_version: u32,
    /// Crate version.
    pub version: String,
    /// Declared neat-core baseline from `neat-core.expected-version`, so an
    /// artefact says which core it was measured against.
    pub neat_core_baseline: String,
    /// Issue that introduced the probe (#40). Later extensions — the #107
    /// facets and gradient error — are tracked by `schema_version`, so this
    /// stays put and a v1 consumer keeps reading the field it knows.
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
    /// Seed the sampling ran under — the artefact is reproducible from it.
    pub seed: u64,
    /// Shape of the probed creature.
    pub creature: CreatureFingerprint,
    /// Per-class aggregates.
    pub by_class: Vec<ClassStats>,
    /// Per-facet aggregates — every axis in [`crate::gene_facets::facet`].
    pub by_facet: Vec<FacetStats>,
    /// Best-performing gene classes by sign agreement.
    pub best_classes: Vec<FacetStats>,
    /// Worst-performing gene classes by sign agreement.
    pub worst_classes: Vec<FacetStats>,
    /// Genes sampled overall.
    pub sampled: usize,
    /// Overall scored genes.
    pub scored: usize,
    /// Overall sign agreements.
    pub sign_agree: usize,
    /// Overall sign agreement percentage.
    pub sign_agree_pct: f64,
    /// Sampled genes whose applied proposal lowered slice MSE.
    pub improved: usize,
    /// `improved / sampled` as a percentage.
    pub improved_pct: f64,
    /// Median relative gradient error over the sample.
    pub grad_rel_error_p50: Option<f64>,
    /// 90th percentile of the relative gradient error.
    pub grad_rel_error_p90: Option<f64>,
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
    /// Minimum scored genes before a facet bucket is ranked best / worst.
    pub facet_min_scored: usize,
    /// How many buckets the best / worst lists carry.
    pub rank_limit: usize,
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
    let creature = load_forward_only_creature(req.creature)?;
    fs::create_dir_all(req.output_dir).map_err(|e| e.to_string())?;

    let lr = calculate_learning_rate(req.config, 0, None);
    let step = effective_step_scale(req.step_scale);
    // A proposal delta is inverted back through this scale to the gradient it
    // implies, so a zero or non-finite one is refused here rather than
    // silently substituted — a substituted scale would report an invented
    // gradient as a measured one.
    let proposal_scale = lr * step;
    if !(proposal_scale.is_finite() && proposal_scale > 0.0) {
        return Err(format!(
            "learning rate × step scale must be positive and finite \
             (learning rate {lr}, step scale {step})"
        ));
    }

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
    let topology = CreatureTopology::of(&creature);
    let bias_pool = eligible_biases(&creature, &report.learning, req.config, &filter);
    let weight_pool =
        eligible_weights(&creature, &report.learning, req.config, &filter, &topology)?;

    let sampled_biases = stratified_sample(&bias_pool, req.sample_biases, &mut rng);
    let sampled_weights = stratified_sample(&weight_pool, req.sample_weights, &mut rng);

    let fd = FdCtx {
        eps: req.fd_eps,
        fd_floor: req.config.plank_constant.max(1e-12),
        proposal_scale,
        training_data: req.training_data,
        max_records: req.max_records,
    };
    let mut genes = Vec::with_capacity(sampled_biases.len() + sampled_weights.len());

    for gene in sampled_biases.iter().chain(sampled_weights.iter()) {
        let attributes = attributes_for(
            &creature,
            &topology,
            &report.neuron_traces,
            gene.attribute_neuron,
        )?;
        genes.push(probe_gene(&creature, gene, &fd, baseline_mse, attributes)?);
    }

    let by_class = aggregate_by_class(&genes);
    let facet_rows: Vec<FacetRow<'_>> = genes.iter().map(facet_row).collect();
    let by_facet = aggregate_facets(&facet_rows);
    let (best_classes, worst_classes) =
        rank_facets(&by_facet, req.facet_min_scored, req.rank_limit);
    let sampled = genes.len();
    let scored: usize = by_class.iter().map(|c| c.scored).sum();
    let sign_agree: usize = by_class.iter().map(|c| c.sign_agree).sum();
    let improved = genes.iter().filter(|g| g.improved).count();
    let errors = sorted_values(genes.iter().filter_map(|g| g.grad_rel_error));

    let summary = GradientCheckSummary {
        schema_version: GRADIENT_CHECK_SCHEMA,
        version: env!("CARGO_PKG_VERSION").to_string(),
        neat_core_baseline: neat_core_baseline(),
        issue: 40,
        baseline_mse,
        records: report.records,
        fd_eps: req.fd_eps,
        step_scale: step,
        learning_rate: lr,
        seed: req.seed,
        creature: CreatureFingerprint::of(&creature),
        by_class,
        by_facet,
        best_classes,
        worst_classes,
        sampled,
        scored,
        sign_agree,
        sign_agree_pct: percentage(sign_agree, scored),
        improved,
        improved_pct: percentage(improved, sampled),
        grad_rel_error_p50: percentile(&errors, 0.50),
        grad_rel_error_p90: percentile(&errors, 0.90),
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
    fs::write(req.output_dir.join("summary.txt"), summary_text(&summary))
        .map_err(|e| e.to_string())?;

    Ok(summary)
}

/// The concise report an unattended run leaves behind (issue #107).
///
/// Written to `<output-dir>/summary.txt` and printed by the CLI: the headline
/// numbers, then the best and worst gene classes with the evidence behind
/// each. Everything here is also in `gradient-check.json` — this is the human
/// end of the same artefact, never a second source of truth.
pub fn summary_text(summary: &GradientCheckSummary) -> String {
    let mut text = format!(
        "gradient-check v{} schema={} neat-core={} seed={} records={} creature={}n/{}s\n\
         sampled={} scored={} signAgree={:.1}% improved={:.1}% gradRelErrorP50={} gradRelErrorP90={}\n",
        summary.version,
        summary.schema_version,
        summary.neat_core_baseline,
        summary.seed,
        summary.records,
        summary.creature.neurons,
        summary.creature.synapses,
        summary.sampled,
        summary.scored,
        summary.sign_agree_pct,
        summary.improved_pct,
        format_optional(summary.grad_rel_error_p50),
        format_optional(summary.grad_rel_error_p90),
    );
    for (label, ranked) in [
        ("best ", &summary.best_classes),
        ("worst", &summary.worst_classes),
    ] {
        if ranked.is_empty() {
            // An empty worst list beside a populated best list means the whole
            // ranking already fitted above — saying "nothing was rankable"
            // there would be a plain untruth.
            let reason = if summary.best_classes.is_empty() {
                "no bucket carried enough scored genes to rank"
            } else {
                "every ranked bucket is already listed above"
            };
            text.push_str(&format!("{label}: {reason}\n"));
        }
        for stats in ranked {
            text.push_str(&format!(
                "{label}: {}={} signAgree={:.1}% improved={:.1}% gradRelErrorP50={} n={}\n",
                stats.facet,
                stats.bucket,
                stats.sign_agree_pct,
                stats.improved_pct,
                format_optional(stats.grad_rel_error_p50),
                stats.scored,
            ));
        }
    }
    text
}

/// Render an optional statistic without pretending a missing one is zero.
fn format_optional(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_string(), |v| format!("{v:.4}"))
}

#[derive(Debug, Clone)]
struct EligibleGene {
    class: GeneClass,
    index: usize,
    /// Neuron the facet attributes are read from — the neuron itself for a
    /// bias, the *target* neuron for a weight.
    attribute_neuron: usize,
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
    /// `learning_rate × step_scale` — inverts a proposal delta back into the
    /// gradient it implies. Never zero: the caller floors it.
    proposal_scale: f64,
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
            attribute_neuron: i,
            current: neuron.bias,
            proposal_delta: delta,
            id: neuron.uuid.clone(),
        });
    }
    out
}

/// Eligible synapse genes.
///
/// A synapse whose target UUID is not a neuron of this creature is a corrupt
/// export — refused here rather than silently dropped, so the sample can never
/// quietly shrink. `PropagateLayout::from_creature` rejects the same export
/// first on any ordinary run, so this is defence in depth on the sampling
/// path, not the primary guard.
fn eligible_weights(
    creature: &CreatureExport,
    signal: &LearningSignal,
    config: &BackpropConfig,
    filter: &PoolFilter<'_>,
    topology: &CreatureTopology,
) -> Result<Vec<EligibleGene>, String> {
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
        let attribute_neuron = topology.neuron_index(&syn.to_uuid).ok_or_else(|| {
            format!(
                "synapse {} targets '{}', which is not a neuron of this creature",
                i, syn.to_uuid
            )
        })?;
        out.push(EligibleGene {
            class,
            index: i,
            attribute_neuron,
            current: syn.weight,
            proposal_delta: delta,
            id: format!("{}→{}", syn.from_uuid, syn.to_uuid),
        });
    }
    Ok(out)
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

/// Probe one gene: central finite difference, then the proposal applied on
/// its own so the row carries what the move actually did (issue #107).
///
/// Three MSE passes per gene — `+ε`, `−ε` and `+Δ` — so a run's cost is
/// `(2 + 3·genes)` passes over the record slice. That is what the sample caps
/// bound.
fn probe_gene(
    creature: &CreatureExport,
    gene: &EligibleGene,
    fd: &FdCtx<'_>,
    baseline_mse: f64,
    attributes: GeneAttributes,
) -> Result<GeneProbeRow, String> {
    let mse_plus = mse_with_gene(creature, gene, gene.current + fd.eps, fd)?;
    let mse_minus = mse_with_gene(creature, gene, gene.current - fd.eps, fd)?;
    let fd_grad = (mse_plus - mse_minus) / (2.0 * fd.eps);
    let applied = mse_with_gene(creature, gene, gene.current + gene.proposal_delta, fd)?;
    Ok(finalize_row(
        gene,
        fd_grad,
        applied - baseline_mse,
        fd,
        attributes,
    ))
}

/// Slice MSE with this one gene set to `value`; the creature is not mutated.
fn mse_with_gene(
    creature: &CreatureExport,
    gene: &EligibleGene,
    value: f64,
    fd: &FdCtx<'_>,
) -> Result<f64, String> {
    let mut mutated = creature.clone();
    if gene.class.is_bias() {
        mutated.neurons[gene.index].bias = value;
    } else {
        mutated.synapses[gene.index].weight = value;
    }
    let mut net = compile_creature(&mutated).map_err(|e| e.to_string())?;
    Ok(compute_mse(&mutated, &mut net, fd.training_data, fd.max_records)?.0)
}

fn finalize_row(
    gene: &EligibleGene,
    fd_grad: f64,
    actual_delta_mse: f64,
    fd: &FdCtx<'_>,
    attributes: GeneAttributes,
) -> GeneProbeRow {
    let scored = fd_grad.is_finite() && fd_grad.abs() >= fd.fd_floor;
    let sign_agree = scored && gene.proposal_delta * fd_grad < 0.0;
    let magnitude_ratio = if scored {
        Some(gene.proposal_delta.abs() / fd_grad.abs())
    } else {
        None
    };
    // A descent step is `Δ = −lr · step · g`, so inverting it recovers the
    // gradient the proposal implies and puts it in the finite difference's
    // units. Clamped proposals show up here as a gradient error, which is
    // exactly what the diagnostic is asked to measure.
    let proposal_grad = -gene.proposal_delta / fd.proposal_scale;
    let grad_abs_error = (proposal_grad - fd_grad).abs();
    // Symmetric relative error: scaled by whichever gradient is larger, so a
    // near-zero denominator cannot inflate the statistic. Only a gene the FD
    // actually scored gets one — an unscored gene has `fd_grad ≈ 0`, which
    // would otherwise contribute a meaningless 1.0 to every distribution.
    let scale = proposal_grad.abs().max(fd_grad.abs());
    let grad_rel_error =
        (scored && scale > 0.0 && scale.is_finite()).then(|| grad_abs_error / scale);
    GeneProbeRow {
        class: gene.class,
        index: gene.index,
        id: gene.id.clone(),
        current: gene.current,
        proposal_delta: gene.proposal_delta,
        fd_grad,
        sign_agree,
        magnitude_ratio,
        attributes,
        proposal_grad,
        grad_abs_error,
        grad_rel_error,
        predicted_delta_mse: fd_grad * gene.proposal_delta,
        actual_delta_mse,
        improved: actual_delta_mse < 0.0,
    }
}

/// View one probe row as a facet row for [`crate::gene_facets`].
fn facet_row(row: &GeneProbeRow) -> FacetRow<'_> {
    FacetRow {
        class: row.class.as_str(),
        gene_kind: row.class.gene_kind(),
        role: row.class.role(),
        attributes: &row.attributes,
        proposal_delta: row.proposal_delta,
        scored: row.magnitude_ratio.is_some(),
        sign_agree: row.sign_agree,
        improved: row.improved,
        magnitude_ratio: row.magnitude_ratio,
        grad_abs_error: row.magnitude_ratio.map(|_| row.grad_abs_error),
        grad_rel_error: row.grad_rel_error,
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
        let ratios = sorted_values(scored_rows.iter().filter_map(|g| g.magnitude_ratio));
        stats.push(ClassStats {
            class: class.as_str().to_string(),
            sampled,
            scored,
            sign_agree,
            sign_agree_pct: percentage(sign_agree, scored),
            magnitude_ratio_p50: percentile(&ratios, 0.50),
            magnitude_ratio_p90: percentile(&ratios, 0.90),
        });
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backprop::BackpropConfig;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn the_neat_core_baseline_is_read_from_the_committed_file() {
        let baseline = neat_core_baseline();
        assert_ne!(
            baseline, "unknown",
            "the committed file must carry a version"
        );
        assert!(
            !baseline.starts_with('#') && baseline.split('.').count() == 3,
            "expected a semver baseline, got '{baseline}'"
        );
    }

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
            facet_min_scored: 5,
            rank_limit: 5,
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
