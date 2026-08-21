//! Accumulate once, then apply the same learning at many step scales.

use crate::backprop::{
    ApplyOptions, BackpropConfig, apply_learnings_with, calculate_learning_rate, count_apply_deltas,
};
use crate::creature_io::{ObservationWidth, load_forward_only_creature};
use crate::mse::compute_mse;
use crate::propagate_layout::accumulate_creature_learning_report;
use crate::validate::TrainedTopology;
use neat_core::compile_creature;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// One applied step-scale in a sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SweepRow {
    /// Step scale used for this apply.
    pub step_scale: f64,
    /// Train-directory MSE after apply.
    pub train_mse: f64,
    /// Optional holdout MSE after apply.
    pub eval_mse: Option<f64>,
    /// Hidden / constant biases that moved.
    pub hidden_biases: usize,
    /// Output biases that moved.
    pub output_biases: usize,
    /// Non-output-target synapses that moved.
    pub hidden_weights: usize,
    /// Output-target synapses that moved.
    pub output_weights: usize,
    /// Relative path of the written candidate.
    pub candidate: String,
}

/// Outcome of [`run_sweep`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SweepSummary {
    /// Crate version.
    pub version: String,
    /// Baseline train MSE.
    pub baseline_train_mse: f64,
    /// Optional baseline holdout MSE.
    pub baseline_eval_mse: Option<f64>,
    /// Records consumed while accumulating.
    pub records: u64,
    /// Per-scale rows.
    pub rows: Vec<SweepRow>,
}

/// Arguments for [`run_sweep`].
pub struct SweepRequest<'a> {
    /// Creature JSON path.
    pub creature: &'a Path,
    /// Training-data directory used for accumulate + train MSE.
    pub training_data: &'a Path,
    /// Optional holdout directory.
    pub eval_data: Option<&'a Path>,
    /// Backprop config.
    pub config: &'a BackpropConfig,
    /// Optional record cap for accumulate and train MSE.
    pub max_records: Option<u64>,
    /// RNG seed.
    pub seed: u64,
    /// Step scales to apply (must be non-empty).
    pub step_scales: &'a [f64],
    /// Apply only output genes.
    pub outputs_only: bool,
    /// Apply only hidden genes.
    pub hidden_only: bool,
    /// Skip train/eval MSE (write candidates only; score with rust_scorer).
    pub skip_mse: bool,
    /// Output directory for candidates + `sweep.json`.
    pub output_dir: &'a Path,
}

/// Accumulate once and write a candidate per step scale.
pub fn run_sweep(req: SweepRequest<'_>) -> Result<SweepSummary, String> {
    if req.step_scales.is_empty() {
        return Err("sweep requires at least one step scale".into());
    }
    if req.outputs_only && req.hidden_only {
        return Err("sweep cannot set both outputs-only and hidden-only".into());
    }
    let incumbent = load_forward_only_creature(req.creature)?;
    let width = ObservationWidth::of(&incumbent)?;
    let topology = TrainedTopology::of(&incumbent);
    fs::create_dir_all(req.output_dir).map_err(|e| e.to_string())?;
    let lr = calculate_learning_rate(req.config, 0, None);
    let (baseline_train_mse, baseline_eval_mse) = if req.skip_mse {
        (0.0, None)
    } else {
        let mut network = compile_creature(&incumbent).map_err(|e| e.to_string())?;
        let (train_mse, _) =
            compute_mse(&incumbent, &mut network, req.training_data, req.max_records)?;
        let eval_mse = match req.eval_data {
            Some(dir) => {
                let mut eval_net = compile_creature(&incumbent).map_err(|e| e.to_string())?;
                Some(compute_mse(&incumbent, &mut eval_net, dir, None)?.0)
            }
            None => None,
        };
        (train_mse, eval_mse)
    };

    let mut rng = StdRng::seed_from_u64(req.seed);
    let mut net = compile_creature(&incumbent).map_err(|e| e.to_string())?;
    let report = accumulate_creature_learning_report(
        &incumbent,
        &mut net,
        req.training_data,
        req.config,
        req.max_records,
        &mut rng,
    )?;

    let candidates_dir = req.output_dir.join("candidates");
    fs::create_dir_all(&candidates_dir).map_err(|e| e.to_string())?;
    let mut rows = Vec::with_capacity(req.step_scales.len());
    for &step_scale in req.step_scales {
        let apply = ApplyOptions {
            step_scale,
            outputs_only: req.outputs_only,
            hidden_only: req.hidden_only,
        };
        let candidate = apply_learnings_with(&incumbent, &report.learning, req.config, lr, apply);
        let deltas = count_apply_deltas(&incumbent, &candidate, req.config.plank_constant);
        let (train_mse, eval_mse) = if req.skip_mse {
            (0.0, None)
        } else {
            let mut cand_net = compile_creature(&candidate).map_err(|e| e.to_string())?;
            let train = compute_mse(
                &candidate,
                &mut cand_net,
                req.training_data,
                req.max_records,
            )?
            .0;
            let eval = match req.eval_data {
                Some(dir) => {
                    let mut eval_net = compile_creature(&candidate).map_err(|e| e.to_string())?;
                    Some(compute_mse(&candidate, &mut eval_net, dir, None)?.0)
                }
                None => None,
            };
            (train, eval)
        };
        // Issue #94: a sweep candidate is a trained creature that reaches
        // disk, so it is gated the same way — once, as it is produced.
        topology.assert_valid(&candidate, &format!("sweep candidate st={step_scale:.8}"))?;
        let name = format!("st{step_scale:.8}.json");
        fs::write(
            candidates_dir.join(&name),
            width.checked_json_pretty(&candidate)?,
        )
        .map_err(|e| e.to_string())?;
        eprintln!(
            "sweep st={step_scale:.8} train_mse={train_mse:.12} eval_mse={} hidden_b={} out_b={} hidden_w={} out_w={}",
            eval_mse
                .map(|v| format!("{v:.12}"))
                .unwrap_or_else(|| "-".into()),
            deltas.hidden_biases,
            deltas.output_biases,
            deltas.hidden_weights,
            deltas.output_weights
        );
        rows.push(SweepRow {
            step_scale,
            train_mse,
            eval_mse,
            hidden_biases: deltas.hidden_biases,
            output_biases: deltas.output_biases,
            hidden_weights: deltas.hidden_weights,
            output_weights: deltas.output_weights,
            candidate: format!("candidates/{name}"),
        });
    }

    let summary = SweepSummary {
        version: env!("CARGO_PKG_VERSION").to_string(),
        baseline_train_mse,
        baseline_eval_mse,
        records: report.records,
        rows,
    };
    fs::write(
        req.output_dir.join("sweep.json"),
        serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn sweep_writes_candidates_and_summary() {
        let dir = tempdir().unwrap();
        let data = dir.path().join("data");
        fs::create_dir_all(&data).unwrap();
        let mut f = fs::File::create(data.join("0.bin")).unwrap();
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
        let scales = [0.01, 0.1];
        let summary = run_sweep(SweepRequest {
            creature: &creature_path,
            training_data: &data,
            eval_data: None,
            config: &cfg,
            max_records: Some(1),
            seed: 1,
            step_scales: &scales,
            outputs_only: false,
            hidden_only: false,
            skip_mse: false,
            output_dir: &out,
        })
        .unwrap();
        assert_eq!(summary.rows.len(), 2);
        assert!(out.join("sweep.json").is_file());
        assert!(out.join("candidates").join("st0.01000000.json").is_file());
    }
}
