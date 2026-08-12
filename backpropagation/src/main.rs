//! Experimental standalone backpropagation CLI.

use clap::{Parser, Subcommand};
use neat_ai_backpropagation::backprop::{ApplyOptions, BackpropConfig};
use neat_ai_backpropagation::compare::{diff_compare_dumps, load_compare_dump, run_compare};
use neat_ai_backpropagation::sweep::{SweepRequest, run_sweep};
use neat_ai_backpropagation::train::{TrainRequest, default_output_dir, run_train};
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
        #[arg(long)]
        max_records: Option<u64>,
        /// Sparse-selection seed.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Learning rate (fixed strategy).
        #[arg(long, default_value_t = 0.01)]
        learning_rate: f64,
        /// Maximum |Δbias| per apply (TS trainDir default is 1).
        #[arg(long, default_value_t = 1.0)]
        maximum_bias_adjustment_scale: f64,
        /// Maximum |Δweight| per apply.
        #[arg(long, default_value_t = 1.0)]
        maximum_weight_adjustment_scale: f64,
        /// Multiply (proposed − current) by this factor before writing.
        #[arg(long, default_value_t = 1.0)]
        step_scale: f64,
        /// Apply only output neurons and synapses that target them.
        #[arg(long, default_value_t = false)]
        outputs_only: bool,
        /// Apply only hidden / constant genes (skip output).
        #[arg(long, default_value_t = false)]
        hidden_only: bool,
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
        #[arg(
            long,
            default_value = "0.000001,0.00001,0.0001,0.0005,0.001,0.002,0.005,0.01"
        )]
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
            learning_rate,
            maximum_bias_adjustment_scale,
            maximum_weight_adjustment_scale,
            step_scale,
            outputs_only,
            hidden_only,
            accept_always,
            max_backtracks,
            scorer,
            output_dir,
        } => {
            let cfg = BackpropConfig {
                learning_rate,
                initial_learning_rate: learning_rate,
                maximum_bias_adjustment_scale,
                maximum_weight_adjustment_scale,
                ..BackpropConfig::default()
            };
            let result = run_train(TrainRequest {
                creature: &creature,
                training_data: &training_data,
                config: &cfg,
                epochs,
                max_records,
                seed,
                output_dir: &output_dir,
                scorer: scorer.as_deref(),
                apply: ApplyOptions {
                    step_scale,
                    outputs_only,
                    hidden_only,
                },
                accept_always,
                max_backtracks,
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
