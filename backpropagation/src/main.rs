//! Experimental standalone backpropagation CLI.

use clap::{Parser, Subcommand, ValueEnum};
use neat_ai_backpropagation::backprop::{ApplyOptions, BackpropConfig, LearningRateStrategy};
use neat_ai_backpropagation::blocks::{BlockPlan, BlockStrategy};
use neat_ai_backpropagation::blockwise::{BlocksRequest, run_blocks};
use neat_ai_backpropagation::compare::{diff_compare_dumps, load_compare_dump, run_compare};
use neat_ai_backpropagation::gradient_check::{
    GradientCheckRequest, run_gradient_check, summary_text,
};
use neat_ai_backpropagation::ladder::{DEFAULT_STEP_SCALE_LADDER_CSV, parse_step_scale_ladder};
use neat_ai_backpropagation::sweep::{SweepRequest, run_sweep};
use neat_ai_backpropagation::targets::{TargetPlan, TargetStrategy};
use neat_ai_backpropagation::train::{
    AcceptanceMode, DEFAULT_MIN_SCORE_IMPROVEMENT, DEFAULT_STEP_SCALE, TrainCreature, TrainRequest,
    default_output_dir, resolve_acceptance, run_train,
};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(name = "neat_ai_backpropagation")]
#[command(about = "Experimental standalone backpropagation for production NEAT-AI creatures")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Sweep's default grid stops at 1% — `train` must not default above it (#39).
const SWEEP_DEFAULT_STEP_SCALES: &str = "0.000001,0.00001,0.0001,0.0005,0.001,0.002,0.005,0.01";

/// Learning-rate strategy selectable from the CLI (mirrors [`LearningRateStrategy`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum LearningRateStrategyArg {
    /// Constant learning rate.
    Fixed,
    /// Multiplicative decay each epoch.
    Decay,
    /// Boost / shrink from epoch-to-epoch MSE feedback.
    Adaptive,
    /// Decay with a periodic warm restart.
    WarmRestart,
}

impl LearningRateStrategyArg {
    /// Map the CLI value onto the library strategy.
    fn to_config(self) -> LearningRateStrategy {
        match self {
            Self::Fixed => LearningRateStrategy::Fixed,
            Self::Decay => LearningRateStrategy::Decay,
            Self::Adaptive => LearningRateStrategy::Adaptive,
            Self::WarmRestart => LearningRateStrategy::WarmRestart,
        }
    }
}

/// What decides accept / rollback, selectable from the CLI (#104).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum AcceptanceModeArg {
    /// Training-slice MSE (the historical behaviour).
    Mse,
    /// `NEAT-AI-scorer` fitness — requires `--scorer`.
    Scorer,
}

impl AcceptanceModeArg {
    /// Map the CLI value onto the library acceptance mode, refusing scorer
    /// settings an MSE run would ignore.
    fn to_config(
        self,
        min_improvement: f64,
        mse_pre_screen: bool,
    ) -> Result<AcceptanceMode, String> {
        resolve_acceptance(self == Self::Scorer, min_improvement, mse_pre_screen)
    }
}

/// Block-selection strategy selectable from the CLI (mirrors [`BlockStrategy`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BlockStrategyArg {
    /// Every gene — the whole-creature apply, kept for parity.
    Global,
    /// One neuron's bias plus every synapse incident to it.
    Neuron,
    /// One neuron plus its neighbours out to `--radius` hops.
    Neighbourhood,
    /// Output neurons and the synapses that reach them.
    OutputHead,
    /// A seeded random connected subgraph.
    Subgraph,
    /// The loudest genes by accumulated proposal magnitude.
    TopGenes,
}

impl BlockStrategyArg {
    /// Map the CLI value onto the library strategy.
    fn to_config(self) -> BlockStrategy {
        match self {
            Self::Global => BlockStrategy::Global,
            Self::Neuron => BlockStrategy::Neuron,
            Self::Neighbourhood => BlockStrategy::Neighbourhood,
            Self::OutputHead => BlockStrategy::OutputHead,
            Self::Subgraph => BlockStrategy::Subgraph,
            Self::TopGenes => BlockStrategy::TopGenes,
        }
    }
}

/// Target-selection strategy selectable from the CLI (mirrors
/// [`TargetStrategy`], issue #108).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TargetStrategyArg {
    /// Uniform random draw over every eligible neuron — the control arm.
    Random,
    /// Highest-ranked accumulated evidence first.
    Evidence,
}

impl TargetStrategyArg {
    /// Map the CLI value onto the library strategy.
    fn to_config(self) -> TargetStrategy {
        match self {
            Self::Random => TargetStrategy::Random,
            Self::Evidence => TargetStrategy::Evidence,
        }
    }
}

/// Default `blocks --strategies` list — every strategy, global first.
const BLOCK_DEFAULT_STRATEGIES: &str = "global,neuron,neighbourhood,output-head,subgraph,top-genes";

/// Build the `train` backprop config from its CLI arguments.
fn train_backprop_config(
    learning_rate: f64,
    strategy: LearningRateStrategyArg,
    learning_rate_decay: f64,
    maximum_bias_adjustment_scale: f64,
    maximum_weight_adjustment_scale: f64,
    normalise_gradients: bool,
) -> BackpropConfig {
    BackpropConfig {
        learning_rate,
        initial_learning_rate: learning_rate,
        learning_rate_strategy: strategy.to_config(),
        learning_rate_decay,
        maximum_bias_adjustment_scale,
        maximum_weight_adjustment_scale,
        normalise_gradients,
        ..BackpropConfig::default()
    }
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Accumulate learning on a creature and write a parity dump.
    Compare {
        /// Creature JSON (UUID-only export).
        creature: PathBuf,
        /// Directory of little-endian f32 `.bin` records.
        training_data: PathBuf,
        /// Max records to consume (omit for the full directory).
        #[arg(long)]
        max_records: Option<u64>,
        /// Sparse-selection seed.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Output JSON path.
        #[arg(long, default_value = "rust-compare.json")]
        out: PathBuf,
    },
    /// Diff two compare dumps (Rust vs TypeScript).
    Diff {
        /// Left-hand dump (typically Rust).
        rust_dump: PathBuf,
        /// Right-hand dump (typically TypeScript).
        other_dump: PathBuf,
        /// Also require matching counts on genes only one side updated
        /// (fails on production IF/MIN/MAX continuation).
        #[arg(long, default_value_t = false)]
        strict: bool,
    },
    /// trainDir-style epochs: apply only when post-apply MSE improves.
    Train {
        /// Creature JSON (UUID-only export).
        creature: PathBuf,
        /// Directory of little-endian f32 `.bin` records.
        training_data: PathBuf,
        /// Epochs to run.
        #[arg(long, default_value_t = 1)]
        epochs: u64,
        /// Max records per epoch / eval (omit for the full directory).
        ///
        /// Honoured as a rate over the whole corpus: every `.bin` file
        /// contributes ceil(file_records × max_records / total_records)
        /// seeded-random records, matching NEAT-AI trainingSampleRate.
        #[arg(long)]
        max_records: Option<u64>,
        /// Sparse-selection and record-sampling seed.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Take each file's leading records for --max-records instead of a
        /// seeded random draw (NEAT-AI disableRandomSamples).
        #[arg(long, default_value_t = false)]
        disable_random_samples: bool,
        /// Initial learning rate (see --learning-rate-strategy).
        #[arg(long, default_value_t = 0.01)]
        learning_rate: f64,
        /// Learning-rate schedule across epochs.
        #[arg(long, value_enum, default_value = "fixed")]
        learning_rate_strategy: LearningRateStrategyArg,
        /// Per-epoch decay factor for the decay / warm-restart strategies.
        #[arg(long, default_value_t = 0.95)]
        learning_rate_decay: f64,
        /// Divide multi-path gradients by sqrt(path count) (NEAT-AI #1872).
        #[arg(long, default_value_t = false)]
        normalise_gradients: bool,
        /// Maximum |Δbias| per apply (TS trainDir default is 1).
        #[arg(long, default_value_t = 1.0)]
        maximum_bias_adjustment_scale: f64,
        /// Maximum |Δweight| per apply.
        #[arg(long, default_value_t = 1.0)]
        maximum_weight_adjustment_scale: f64,
        /// Multiply (proposed − current) by this factor before writing.
        #[arg(long, default_value_t = DEFAULT_STEP_SCALE)]
        step_scale: f64,
        /// Comma-separated step scales to score as a ladder (#106).
        ///
        /// Requires `--acceptance scorer`. The epoch's one accumulation is
        /// applied at every rung, the whole grid is scored in a single
        /// `rust_scorer` call, and the best scorer improvement wins —
        /// superseding the `--max-backtracks` halving search. Pass the flag
        /// without a value for the default grid.
        #[arg(long, num_args = 0..=1, default_missing_value = DEFAULT_STEP_SCALE_LADDER_CSV)]
        step_scale_ladder: Option<String>,
        /// Apply only output neurons and synapses that target them.
        #[arg(long, default_value_t = false)]
        outputs_only: bool,
        /// Apply only hidden / constant genes (skip output).
        #[arg(long, default_value_t = false)]
        hidden_only: bool,
        /// What decides accept / rollback: slice MSE, or the scorer (#104).
        ///
        /// `scorer` requires `--scorer` and makes `NEAT-AI-scorer` the judge
        /// during the loop — MSE stays a journalled diagnostic.
        #[arg(long, value_enum, default_value = "mse")]
        acceptance: AcceptanceModeArg,
        /// Minimum scorer gain to keep a candidate under `--acceptance scorer`.
        #[arg(long, default_value_t = DEFAULT_MIN_SCORE_IMPROVEMENT)]
        min_score_improvement: f64,
        /// Under `--acceptance scorer`, drop a candidate whose slice MSE did
        /// not fall instead of paying for a scorer run.
        #[arg(long, default_value_t = false)]
        mse_pre_screen: bool,
        /// Keep the applied creature even if slice MSE rose.
        #[arg(long, default_value_t = false)]
        accept_always: bool,
        /// Backtracking line search (#38): halvings of step-scale to retry a
        /// rejected apply with, reusing the epoch's accumulated learning.
        #[arg(long, default_value_t = 6)]
        max_backtracks: u32,
        /// Optional `rust_scorer` binary for before/after score.
        #[arg(long)]
        scorer: Option<PathBuf>,
        /// Output directory for `best.json` and `journal.jsonl`.
        #[arg(long, default_value_os_t = default_output_dir())]
        output_dir: PathBuf,
        /// NEAT-AI traceStore directory for CreatureTrace artifacts.
        ///
        /// An epoch that lowered the best MSE writes `best-trace.json` beside
        /// `best.json`; one that did not writes
        /// `<store>/failed/epoch-<N>.json`, matching NEAT-AI's failed-candidate
        /// store. Omit to write no traces.
        #[arg(long)]
        trace_store: Option<PathBuf>,
    },
    /// Accumulate once, then apply the same learning at many step scales.
    Sweep {
        /// Creature JSON (UUID-only export).
        creature: PathBuf,
        /// Directory of little-endian f32 `.bin` records (accumulate + train MSE).
        training_data: PathBuf,
        /// Optional holdout directory.
        #[arg(long)]
        eval_dir: Option<PathBuf>,
        /// Max records for accumulate / train MSE.
        #[arg(long)]
        max_records: Option<u64>,
        /// Sparse-selection seed.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Learning rate (fixed strategy).
        #[arg(long, default_value_t = 0.01)]
        learning_rate: f64,
        /// Maximum |Δbias| per apply.
        #[arg(long, default_value_t = 1.0)]
        maximum_bias_adjustment_scale: f64,
        /// Maximum |Δweight| per apply.
        #[arg(long, default_value_t = 1.0)]
        maximum_weight_adjustment_scale: f64,
        /// Comma-separated step scales.
        #[arg(long, default_value = SWEEP_DEFAULT_STEP_SCALES)]
        step_scales: String,
        /// Apply only output neurons and synapses that target them.
        #[arg(long, default_value_t = false)]
        outputs_only: bool,
        /// Apply only hidden / constant genes (skip output).
        #[arg(long, default_value_t = false)]
        hidden_only: bool,
        /// Skip train/eval MSE and only write candidates.
        #[arg(long, default_value_t = false)]
        skip_mse: bool,
        /// Output directory for `sweep.json` and `candidates/`.
        #[arg(long, default_value_os_t = default_output_dir())]
        output_dir: PathBuf,
    },
    /// Accumulate once, then apply that learning one region at a time (#105).
    Blocks {
        /// Creature JSON (UUID-only export).
        creature: PathBuf,
        /// Directory of little-endian f32 `.bin` records.
        training_data: PathBuf,
        /// Max records for the accumulation pass and the MSE checks.
        #[arg(long)]
        max_records: Option<u64>,
        /// Sparse-selection and random-subgraph seed.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Learning rate (fixed strategy).
        #[arg(long, default_value_t = 0.01)]
        learning_rate: f64,
        /// Maximum |Δbias| per apply.
        #[arg(long, default_value_t = 1.0)]
        maximum_bias_adjustment_scale: f64,
        /// Maximum |Δweight| per apply.
        #[arg(long, default_value_t = 1.0)]
        maximum_weight_adjustment_scale: f64,
        /// Multiply (proposed − current) by this factor before writing.
        #[arg(long, default_value_t = DEFAULT_STEP_SCALE)]
        step_scale: f64,
        /// Block strategies to generate, in order.
        #[arg(long, value_enum, value_delimiter = ',', default_value = BLOCK_DEFAULT_STRATEGIES)]
        strategies: Vec<BlockStrategyArg>,
        /// Focus neurons per neuron / neighbourhood / subgraph strategy.
        #[arg(long, default_value_t = 4)]
        blocks_per_strategy: usize,
        /// Hops around the focus neuron for the neighbourhood strategy.
        #[arg(long, default_value_t = 1)]
        radius: usize,
        /// Neurons per random-subgraph walk.
        #[arg(long, default_value_t = 8)]
        subgraph_size: usize,
        /// Genes kept by the top-genes strategy.
        #[arg(long, default_value_t = 32)]
        top_genes: usize,
        /// How focus targets are drawn: ranked evidence, or uniform random
        /// (issue #108).
        #[arg(long, value_enum, default_value_t = TargetStrategyArg::Evidence)]
        target_selection: TargetStrategyArg,
        /// Share of every focus draw reserved for the uniform random control
        /// arm, 0.0–1.0.
        #[arg(long, default_value_t = 0.0)]
        random_control_fraction: f64,
        /// Skip every MSE pass (write and score candidates only).
        #[arg(long, default_value_t = false)]
        skip_mse: bool,
        /// Optional `rust_scorer` binary — scores the baseline and each block.
        #[arg(long)]
        scorer: Option<PathBuf>,
        /// Minimum scorer gain that counts as a win. Needs `--scorer`.
        #[arg(long, default_value_t = DEFAULT_MIN_SCORE_IMPROVEMENT)]
        min_score_improvement: f64,
        /// Output directory for `blocks.json` and `candidates/`.
        #[arg(long, default_value_os_t = default_output_dir())]
        output_dir: PathBuf,
    },
    /// Compare proposal Δ to finite-difference ∂MSE/∂gene (issue #40).
    GradientCheck {
        /// Creature JSON (UUID-only export).
        creature: PathBuf,
        /// Directory of little-endian f32 `.bin` records.
        training_data: PathBuf,
        /// Max records for accumulate and FD MSE.
        #[arg(long)]
        max_records: Option<u64>,
        /// Sparse-selection / sampling seed.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Learning rate (fixed strategy).
        #[arg(long, default_value_t = 0.01)]
        learning_rate: f64,
        /// Maximum |Δbias| per propose.
        #[arg(long, default_value_t = 1.0)]
        maximum_bias_adjustment_scale: f64,
        /// Maximum |Δweight| per propose.
        #[arg(long, default_value_t = 1.0)]
        maximum_weight_adjustment_scale: f64,
        /// Step scale applied to (proposed − current).
        #[arg(long, default_value_t = 1.0)]
        step_scale: f64,
        /// Max bias genes to sample (stratified by class).
        #[arg(long, default_value_t = 50)]
        sample_biases: usize,
        /// Max weight genes to sample (stratified by class).
        #[arg(long, default_value_t = 50)]
        sample_weights: usize,
        /// Central finite-difference ε.
        #[arg(long, default_value_t = 1e-4)]
        fd_eps: f64,
        /// Restrict eligible pool to output genes.
        #[arg(long, default_value_t = false)]
        outputs_only: bool,
        /// Restrict eligible pool to hidden genes.
        #[arg(long, default_value_t = false)]
        hidden_only: bool,
        /// Minimum scored genes before a facet bucket is ranked (issue #107).
        #[arg(long, default_value_t = 5)]
        facet_min_scored: usize,
        /// How many buckets the best / worst lists carry (issue #107).
        #[arg(long, default_value_t = 5)]
        rank_limit: usize,
        /// Output directory for `gradient-check.json` and `genes.jsonl`.
        #[arg(long, default_value_os_t = default_output_dir())]
        output_dir: PathBuf,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Compare {
            creature,
            training_data,
            max_records,
            seed,
            out,
        } => {
            let cfg = BackpropConfig::default();
            let dump = run_compare(&creature, &training_data, &cfg, max_records, seed, &out)?;
            eprintln!(
                "compare: records={} mse={:.12} wrote {}",
                dump.records,
                dump.mse,
                out.display()
            );
            Ok(())
        }
        Commands::Diff {
            rust_dump,
            other_dump,
            strict,
        } => {
            let rust = load_compare_dump(&rust_dump)?;
            let other = load_compare_dump(&other_dump)?;
            let report = diff_compare_dumps(&rust, &other, strict)?;
            eprintln!(
                "overlap neurons={} synapses={}  rust-only neurons={} synapses={}  ts-only neurons={} synapses={}",
                report.overlap_neurons,
                report.overlap_synapses,
                report.rust_only_neurons,
                report.rust_only_synapses,
                report.other_only_neurons,
                report.other_only_synapses
            );
            if report.mismatches.is_empty() {
                eprintln!(
                    "parity ok: {} records, mse rust={:.12} other={:.12}",
                    rust.records, rust.mse, other.mse
                );
                Ok(())
            } else {
                for m in &report.mismatches {
                    eprintln!(
                        "MISMATCH {} rust={:.12} other={:.12} Δ={:.6e}",
                        m.path,
                        m.left,
                        m.right,
                        (m.left - m.right).abs()
                    );
                }
                Err(format!("{} field(s) diverged", report.mismatches.len()))
            }
        }
        Commands::Train {
            creature,
            training_data,
            epochs,
            max_records,
            seed,
            disable_random_samples,
            learning_rate,
            learning_rate_strategy,
            learning_rate_decay,
            normalise_gradients,
            maximum_bias_adjustment_scale,
            maximum_weight_adjustment_scale,
            step_scale,
            step_scale_ladder,
            outputs_only,
            hidden_only,
            acceptance,
            min_score_improvement,
            mse_pre_screen,
            accept_always,
            max_backtracks,
            scorer,
            output_dir,
            trace_store,
        } => {
            let cfg = train_backprop_config(
                learning_rate,
                learning_rate_strategy,
                learning_rate_decay,
                maximum_bias_adjustment_scale,
                maximum_weight_adjustment_scale,
                normalise_gradients,
            );
            // Parsed here, validated by the library — one gate every caller
            // (CLI and C ABI alike) goes through.
            let ladder = match step_scale_ladder.as_deref() {
                Some(raw) => parse_step_scale_ladder(raw)?,
                None => Vec::new(),
            };
            let result = run_train(TrainRequest {
                creature: TrainCreature::Path(&creature),
                training_data: &training_data,
                config: &cfg,
                epochs,
                max_records,
                seed,
                disable_random_samples,
                output_dir: &output_dir,
                scorer: scorer.as_deref(),
                apply: ApplyOptions {
                    step_scale,
                    outputs_only,
                    hidden_only,
                },
                acceptance: acceptance.to_config(min_score_improvement, mse_pre_screen)?,
                accept_always,
                max_backtracks,
                step_scale_ladder: &ladder,
                trace_store: trace_store.as_deref(),
            })?;
            eprintln!(
                "train: baseline_mse={:.12} best_mse={:.12} accepted_epochs={}",
                result.baseline_mse, result.best_mse, result.accepted_epochs
            );
            if let (Some(b), Some(a)) = (&result.baseline_score, &result.best_score) {
                eprintln!(
                    "scorer: baseline={:.12} best={:.12} Δ={:+.6e}",
                    b.score,
                    a.score,
                    a.score - b.score
                );
            }
            Ok(())
        }
        Commands::Sweep {
            creature,
            training_data,
            eval_dir,
            max_records,
            seed,
            learning_rate,
            maximum_bias_adjustment_scale,
            maximum_weight_adjustment_scale,
            step_scales,
            outputs_only,
            hidden_only,
            skip_mse,
            output_dir,
        } => {
            let scales = parse_step_scales(&step_scales)?;
            let cfg = BackpropConfig {
                learning_rate,
                initial_learning_rate: learning_rate,
                maximum_bias_adjustment_scale,
                maximum_weight_adjustment_scale,
                ..BackpropConfig::default()
            };
            let summary = run_sweep(SweepRequest {
                creature: &creature,
                training_data: &training_data,
                eval_data: eval_dir.as_deref(),
                config: &cfg,
                max_records,
                seed,
                step_scales: &scales,
                outputs_only,
                hidden_only,
                skip_mse,
                output_dir: &output_dir,
            })?;
            eprintln!(
                "sweep: records={} baseline_train_mse={:.12} wrote {}",
                summary.records,
                summary.baseline_train_mse,
                output_dir.join("sweep.json").display()
            );
            Ok(())
        }
        Commands::Blocks {
            creature,
            training_data,
            max_records,
            seed,
            learning_rate,
            maximum_bias_adjustment_scale,
            maximum_weight_adjustment_scale,
            step_scale,
            strategies,
            blocks_per_strategy,
            radius,
            subgraph_size,
            top_genes,
            target_selection,
            random_control_fraction,
            skip_mse,
            scorer,
            min_score_improvement,
            output_dir,
        } => {
            let cfg = BackpropConfig {
                learning_rate,
                initial_learning_rate: learning_rate,
                maximum_bias_adjustment_scale,
                maximum_weight_adjustment_scale,
                ..BackpropConfig::default()
            };
            let plan = BlockPlan {
                strategies: strategies
                    .into_iter()
                    .map(BlockStrategyArg::to_config)
                    .collect(),
                blocks_per_strategy,
                radius,
                subgraph_size,
                top_genes,
                targets: TargetPlan {
                    strategy: target_selection.to_config(),
                    random_control_fraction,
                },
            };
            let summary = run_blocks(BlocksRequest {
                creature: &creature,
                training_data: &training_data,
                config: &cfg,
                max_records,
                seed,
                step_scale,
                plan: &plan,
                skip_mse,
                scorer: scorer.as_deref(),
                min_score_improvement,
                output_dir: &output_dir,
            })?;
            eprintln!(
                "blocks: records={} planned={} written={} unmoved={} dropped={} wrote {}",
                summary.records,
                summary.candidates.len(),
                summary.written().len(),
                summary.unmoved_blocks,
                summary.dropped_empty_blocks + summary.dropped_duplicate_blocks,
                output_dir.join("blocks.json").display()
            );
            for winner in summary.winners() {
                eprintln!(
                    "  win {} strategy={:?} score_delta={:+.6e} genes={}",
                    winner.label,
                    winner.strategy,
                    winner.score_delta.unwrap_or_default(),
                    winner.neurons.len() + winner.synapses.len()
                );
            }
            if let Some(comparison) = &summary.selection_comparison {
                for arm in [&comparison.evidence, &comparison.random_control] {
                    eprintln!(
                        "  arm {:?}: scored={} wins={} scorer_seconds={:.1} wins/h={} gain/h={}",
                        arm.source,
                        arm.candidates_scored,
                        arm.wins,
                        arm.scorer_seconds,
                        arm.wins_per_hour
                            .map_or_else(|| "-".into(), |v| format!("{v:.3}")),
                        arm.score_gain_per_hour
                            .map_or_else(|| "-".into(), |v| format!("{v:.6e}")),
                    );
                }
            }
            Ok(())
        }
        Commands::GradientCheck {
            creature,
            training_data,
            max_records,
            seed,
            learning_rate,
            maximum_bias_adjustment_scale,
            maximum_weight_adjustment_scale,
            step_scale,
            sample_biases,
            sample_weights,
            fd_eps,
            outputs_only,
            hidden_only,
            facet_min_scored,
            rank_limit,
            output_dir,
        } => {
            let cfg = BackpropConfig {
                learning_rate,
                initial_learning_rate: learning_rate,
                maximum_bias_adjustment_scale,
                maximum_weight_adjustment_scale,
                ..BackpropConfig::default()
            };
            let summary = run_gradient_check(GradientCheckRequest {
                creature: &creature,
                training_data: &training_data,
                config: &cfg,
                max_records,
                seed,
                sample_biases,
                sample_weights,
                fd_eps,
                step_scale,
                outputs_only,
                hidden_only,
                facet_min_scored,
                rank_limit,
                output_dir: &output_dir,
            })?;
            // The concise report an unattended run reads back (issue #107);
            // the same text is written to `<output-dir>/summary.txt`.
            eprint!("{}", summary_text(&summary));
            eprintln!(
                "gradient-check: wrote {}",
                output_dir.join("gradient-check.json").display()
            );
            for c in &summary.by_class {
                eprintln!(
                    "  {}: sampled={} scored={} sign_agree={:.1}%",
                    c.class, c.sampled, c.scored, c.sign_agree_pct
                );
            }
            Ok(())
        }
    }
}

fn parse_step_scales(raw: &str) -> Result<Vec<f64>, String> {
    let mut scales = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: f64 = trimmed
            .parse()
            .map_err(|e| format!("invalid step scale '{trimmed}': {e}"))?;
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("step scale must be positive and finite: {trimmed}"));
        }
        scales.push(value);
    }
    if scales.is_empty() {
        return Err("no step scales provided".into());
    }
    Ok(scales)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_ai_backpropagation::train::ScorerAcceptance;

    /// Parse a bare `train` invocation and hand back its arguments.
    fn parse_train(extra: &[&str]) -> Commands {
        let mut argv = vec!["neat_ai_backpropagation", "train", "creature.json", "data"];
        argv.extend_from_slice(extra);
        Cli::parse_from(argv).command
    }

    #[test]
    fn gradient_check_ranking_defaults_are_conservative() {
        let Commands::GradientCheck {
            facet_min_scored,
            rank_limit,
            sample_biases,
            sample_weights,
            ..
        } = Cli::parse_from([
            "neat_ai_backpropagation",
            "gradient-check",
            "creature.json",
            "data",
        ])
        .command
        else {
            panic!("expected gradient-check");
        };
        assert_eq!(
            facet_min_scored, 5,
            "a bucket needs real evidence before it is ranked"
        );
        assert_eq!(rank_limit, 5);
        // The caps are what bound an unattended production run.
        assert_eq!(sample_biases, 50);
        assert_eq!(sample_weights, 50);
    }

    #[test]
    fn train_step_scale_defaults_within_sweep_grid() {
        let Commands::Train { step_scale, .. } = parse_train(&[]) else {
            panic!("expected train");
        };
        let grid = parse_step_scales(SWEEP_DEFAULT_STEP_SCALES).unwrap();
        let ceiling = grid.iter().copied().fold(f64::MIN, f64::max);
        // #39: the old 1.0 default was 100× sweep's ceiling and overshot.
        assert!(
            step_scale <= ceiling,
            "train default step scale {step_scale} exceeds sweep ceiling {ceiling}"
        );
        assert!((step_scale - DEFAULT_STEP_SCALE).abs() < 1e-12);
    }

    #[test]
    fn train_clamps_default_to_one_not_ten() {
        let Commands::Train {
            maximum_bias_adjustment_scale,
            maximum_weight_adjustment_scale,
            ..
        } = parse_train(&[])
        else {
            panic!("expected train");
        };
        // The ±10 library default mirrors the TS compare harness; the trainer
        // must not inherit it (#39).
        assert!((maximum_bias_adjustment_scale - 1.0).abs() < 1e-12);
        assert!((maximum_weight_adjustment_scale - 1.0).abs() < 1e-12);
        assert!(
            (BackpropConfig::default().maximum_bias_adjustment_scale - 10.0).abs() < 1e-12,
            "compare parity default must stay at the TS value"
        );
    }

    #[test]
    fn train_learning_rate_defaults_to_fixed_schedule() {
        let Commands::Train {
            learning_rate,
            learning_rate_strategy,
            normalise_gradients,
            ..
        } = parse_train(&[])
        else {
            panic!("expected train");
        };
        assert!((learning_rate - 0.01).abs() < 1e-12);
        assert_eq!(learning_rate_strategy, LearningRateStrategyArg::Fixed);
        assert!(!normalise_gradients);
    }

    #[test]
    fn train_accepts_schedule_and_normalisation_flags() {
        let Commands::Train {
            learning_rate_strategy,
            learning_rate_decay,
            normalise_gradients,
            ..
        } = parse_train(&[
            "--learning-rate-strategy",
            "warm-restart",
            "--learning-rate-decay",
            "0.5",
            "--normalise-gradients",
        ])
        else {
            panic!("expected train");
        };
        assert_eq!(learning_rate_strategy, LearningRateStrategyArg::WarmRestart);
        assert!((learning_rate_decay - 0.5).abs() < 1e-12);
        assert!(normalise_gradients);
    }

    #[test]
    fn train_config_carries_schedule_and_normalisation() {
        let cfg = train_backprop_config(0.2, LearningRateStrategyArg::Decay, 0.5, 1.0, 2.0, true);
        assert_eq!(cfg.learning_rate_strategy, LearningRateStrategy::Decay);
        assert!((cfg.initial_learning_rate - 0.2).abs() < 1e-12);
        assert!((cfg.learning_rate_decay - 0.5).abs() < 1e-12);
        assert!((cfg.maximum_bias_adjustment_scale - 1.0).abs() < 1e-12);
        assert!((cfg.maximum_weight_adjustment_scale - 2.0).abs() < 1e-12);
        assert!(cfg.normalise_gradients);
    }

    /// `--max-records` defaults to a seeded *random* draw (#77) — the flag has
    /// to be asked for, and the bridge relies on that default.
    #[test]
    fn record_sampling_is_random_unless_disabled() {
        let Commands::Train {
            disable_random_samples,
            seed,
            ..
        } = parse_train(&[])
        else {
            panic!("expected train");
        };
        assert!(!disable_random_samples);
        assert_eq!(seed, 1);

        let Commands::Train {
            disable_random_samples,
            seed,
            ..
        } = parse_train(&["--disable-random-samples", "--seed", "42"])
        else {
            panic!("expected train");
        };
        assert!(disable_random_samples);
        assert_eq!(seed, 42);
    }

    /// MSE acceptance stays the default, so an existing `train` invocation is
    /// unchanged by #104 — the scorer gate has to be asked for.
    #[test]
    fn acceptance_defaults_to_mse_with_a_conservative_epsilon() {
        let Commands::Train {
            acceptance,
            min_score_improvement,
            mse_pre_screen,
            ..
        } = parse_train(&[])
        else {
            panic!("expected train");
        };
        assert_eq!(acceptance, AcceptanceModeArg::Mse);
        assert!((min_score_improvement - DEFAULT_MIN_SCORE_IMPROVEMENT).abs() < 1e-18);
        assert!(!mse_pre_screen);
        assert_eq!(
            acceptance
                .to_config(min_score_improvement, mse_pre_screen)
                .unwrap(),
            AcceptanceMode::Mse
        );
    }

    /// A scorer knob on an MSE run is a misconfiguration, not a no-op — the
    /// run would otherwise look configured and quietly judge on MSE (#104).
    #[test]
    fn scorer_settings_on_an_mse_run_are_refused() {
        let err = AcceptanceModeArg::Mse
            .to_config(DEFAULT_MIN_SCORE_IMPROVEMENT, true)
            .unwrap_err();
        assert!(err.contains("msePreScreen"), "unexpected error: {err}");

        let err = AcceptanceModeArg::Mse.to_config(0.5, false).unwrap_err();
        assert!(
            err.contains("minScoreImprovement"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn scorer_acceptance_carries_its_epsilon_and_pre_screen() {
        let Commands::Train {
            acceptance,
            min_score_improvement,
            mse_pre_screen,
            ..
        } = parse_train(&[
            "--acceptance",
            "scorer",
            "--min-score-improvement",
            "0.005",
            "--mse-pre-screen",
        ])
        else {
            panic!("expected train");
        };
        assert_eq!(acceptance, AcceptanceModeArg::Scorer);
        assert_eq!(
            acceptance
                .to_config(min_score_improvement, mse_pre_screen)
                .unwrap(),
            AcceptanceMode::Scorer(ScorerAcceptance {
                min_improvement: 0.005,
                mse_pre_screen: true,
            })
        );
    }

    /// Parse a bare `blocks` invocation and hand back its arguments.
    fn parse_blocks(extra: &[&str]) -> Commands {
        let mut argv = vec!["neat_ai_backpropagation", "blocks", "creature.json", "data"];
        argv.extend_from_slice(extra);
        Cli::parse_from(argv).command
    }

    /// The default plan generates every strategy — including `global`, so a
    /// blockwise run always carries the whole-creature apply to compare
    /// against (#105).
    #[test]
    fn blocks_defaults_generate_every_strategy_including_global() {
        let Commands::Blocks {
            strategies,
            step_scale,
            blocks_per_strategy,
            radius,
            subgraph_size,
            top_genes,
            target_selection,
            random_control_fraction,
            skip_mse,
            min_score_improvement,
            ..
        } = parse_blocks(&[])
        else {
            panic!("expected blocks");
        };
        assert_eq!(strategies[0], BlockStrategyArg::Global);
        assert_eq!(strategies.len(), 6);
        assert!((step_scale - DEFAULT_STEP_SCALE).abs() < 1e-12);
        assert_eq!(blocks_per_strategy, 4);
        assert_eq!(radius, 1);
        assert_eq!(subgraph_size, 8);
        assert_eq!(top_genes, 32);
        assert!(!skip_mse);
        assert!((min_score_improvement - DEFAULT_MIN_SCORE_IMPROVEMENT).abs() < 1e-18);
        // Evidence-driven targets by default, with the random control arm off
        // until it is asked for (issue #108).
        assert_eq!(target_selection, TargetStrategyArg::Evidence);
        assert!(random_control_fraction.abs() < 1e-12);

        // The parsed list must survive the mapping onto the library plan.
        let plan = BlockPlan {
            strategies: strategies
                .into_iter()
                .map(BlockStrategyArg::to_config)
                .collect(),
            blocks_per_strategy,
            radius,
            subgraph_size,
            top_genes,
            targets: TargetPlan {
                strategy: target_selection.to_config(),
                random_control_fraction,
            },
        };
        plan.validate().unwrap();
        assert_eq!(plan.strategies, BlockPlan::default().strategies);
    }

    #[test]
    fn blocks_accepts_a_narrowed_strategy_list() {
        let Commands::Blocks {
            strategies,
            blocks_per_strategy,
            radius,
            ..
        } = parse_blocks(&[
            "--strategies",
            "neuron,neighbourhood",
            "--blocks-per-strategy",
            "12",
            "--radius",
            "2",
        ])
        else {
            panic!("expected blocks");
        };
        assert_eq!(
            strategies
                .into_iter()
                .map(BlockStrategyArg::to_config)
                .collect::<Vec<_>>(),
            vec![BlockStrategy::Neuron, BlockStrategy::Neighbourhood]
        );
        assert_eq!(blocks_per_strategy, 12);
        assert_eq!(radius, 2);
    }

    /// A plan that could only produce nothing is refused, not run empty.
    #[test]
    fn blocks_refuses_a_plan_that_generates_nothing() {
        let Commands::Blocks {
            strategies,
            radius,
            subgraph_size,
            top_genes,
            ..
        } = parse_blocks(&["--blocks-per-strategy", "0"])
        else {
            panic!("expected blocks");
        };
        let plan = BlockPlan {
            strategies: strategies
                .into_iter()
                .map(BlockStrategyArg::to_config)
                .collect(),
            blocks_per_strategy: 0,
            radius,
            subgraph_size,
            top_genes,
            targets: TargetPlan::default(),
        };
        assert!(plan.validate().is_err());
    }

    /// The ladder is off unless it is asked for, and the bare flag selects the
    /// default grid (#106).
    #[test]
    fn the_step_scale_ladder_is_opt_in_with_a_default_grid() {
        let Commands::Train {
            step_scale_ladder, ..
        } = parse_train(&[])
        else {
            panic!("expected train");
        };
        assert_eq!(step_scale_ladder, None);

        let Commands::Train {
            step_scale_ladder, ..
        } = parse_train(&["--step-scale-ladder"])
        else {
            panic!("expected train");
        };
        assert_eq!(
            parse_step_scale_ladder(&step_scale_ladder.expect("bare flag takes the default grid"))
                .unwrap(),
            parse_step_scale_ladder(DEFAULT_STEP_SCALE_LADDER_CSV).unwrap()
        );

        let Commands::Train {
            step_scale_ladder, ..
        } = parse_train(&["--step-scale-ladder", "0.002,0.02"])
        else {
            panic!("expected train");
        };
        assert_eq!(
            parse_step_scale_ladder(&step_scale_ladder.expect("explicit grid")).unwrap(),
            vec![0.002, 0.02]
        );
    }

    #[test]
    fn invalid_step_scales_are_rejected() {
        assert!(parse_step_scales("0.01,-1").is_err());
        assert!(parse_step_scales(" ").is_err());
        assert_eq!(parse_step_scales("0.01, 0.5").unwrap(), vec![0.01, 0.5]);
    }
}
