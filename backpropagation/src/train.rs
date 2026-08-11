//! `trainDir`-style epoch loop: accumulate → apply if MSE improved → else rollback.

use crate::backprop::{
    ApplyOptions, BackpropConfig, apply_learnings_with, calculate_learning_rate, count_apply_deltas,
};
use crate::mse::compute_mse;
use crate::propagate_layout::accumulate_creature_learning_report;
use crate::scorer::{ScoreResult, score_creature};
use neat_core::{CreatureExport, compile_creature, creature_to_json_pretty, parse_creature_json};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Minimum score improvement treated as a production accept (`rust_scorer`).
pub const MIN_SCORE_IMPROVEMENT: f64 = 1e-6;

/// Journal header written at the start of a train run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainJournalHeader {
    /// Discriminator.
    pub kind: String,
    /// Crate version (`CARGO_PKG_VERSION`).
    pub version: String,
    /// Seed used for sparse selection.
    pub seed: u64,
    /// Requested epoch count.
    pub epochs: u64,
    /// Optional record cap.
    pub max_records: Option<u64>,
    /// Learning rate used for apply.
    pub learning_rate: f64,
    /// Apply step scale.
    pub step_scale: f64,
    /// Whether only output genes were written.
    pub outputs_only: bool,
}

/// One epoch line in the journal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainEpochRecord {
    /// Discriminator.
    pub kind: String,
    /// 1-based epoch index.
    pub epoch: u64,
    /// Forward MSE of the incumbent before apply.
    pub before_mse: f64,
    /// Forward MSE of the candidate after apply.
    pub after_mse: f64,
    /// Records used this epoch.
    pub records: u64,
    /// Whether the candidate was kept.
    pub accepted: bool,
    /// Hidden / constant biases that moved.
    pub hidden_biases: usize,
    /// Output biases that moved.
    pub output_biases: usize,
    /// Non-output-target synapses that moved.
    pub hidden_weights: usize,
    /// Output-target synapses that moved.
    pub output_weights: usize,
}

/// Outcome of [`run_train`].
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// Best creature after the run.
    pub creature: CreatureExport,
    /// Baseline MSE (incumbent, before any apply).
    pub baseline_mse: f64,
    /// Best post-apply (or baseline if nothing accepted) MSE.
    pub best_mse: f64,
    /// Epochs that accepted an apply.
    pub accepted_epochs: u64,
    /// Optional baseline scorer result.
    pub baseline_score: Option<ScoreResult>,
    /// Optional best-creature scorer result.
    pub best_score: Option<ScoreResult>,
}

/// Arguments for [`run_train`].
pub struct TrainRequest<'a> {
    /// Creature JSON path.
    pub creature: &'a Path,
    /// Training-data directory.
    pub training_data: &'a Path,
    /// Backprop config.
    pub config: &'a BackpropConfig,
    /// Epochs to run.
    pub epochs: u64,
    /// Optional record cap (same cap used for accumulate and eval MSE).
    pub max_records: Option<u64>,
    /// RNG seed.
    pub seed: u64,
    /// Output directory for journal + best creature.
    pub output_dir: &'a Path,
    /// Optional `rust_scorer` binary. When set, scores baseline and best.
    pub scorer: Option<&'a Path>,
    /// Apply options (step scale / output-only).
    pub apply: ApplyOptions,
    /// When true, keep the applied creature even if slice MSE rose (for
    /// full-corpus scorer checks). Default false = MSE rollback.
    pub accept_always: bool,
}

/// Run the experimental trainer and write `journal.jsonl` + `best.json`.
pub fn run_train(req: TrainRequest<'_>) -> Result<TrainResult, String> {
    fs::create_dir_all(req.output_dir).map_err(|e| e.to_string())?;
    let text = fs::read_to_string(req.creature).map_err(|e| e.to_string())?;
    let mut incumbent = parse_creature_json(&text).map_err(|e| e.to_string())?;
    if !incumbent.forward_only {
        return Err(
            "this trainer supports forward-only creatures only (no re-entrant / recurrent graphs)"
                .into(),
        );
    }
    let lr = calculate_learning_rate(req.config, 0, None);

    let mut network = compile_creature(&incumbent).map_err(|e| e.to_string())?;
    let (baseline_mse, _) =
        compute_mse(&incumbent, &mut network, req.training_data, req.max_records)?;

    let journal_path = req.output_dir.join("journal.jsonl");
    let header = TrainJournalHeader {
        kind: "runHeader".into(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        seed: req.seed,
        epochs: req.epochs,
        max_records: req.max_records,
        learning_rate: lr,
        step_scale: req.apply.step_scale,
        outputs_only: req.apply.outputs_only,
    };
    let mut journal = String::new();
    journal.push_str(&serde_json::to_string(&header).map_err(|e| e.to_string())?);
    journal.push('\n');

    let mut best_mse = baseline_mse;
    let mut accepted_epochs = 0u64;
    let mut rng = StdRng::seed_from_u64(req.seed);

    for epoch in 1..=req.epochs {
        let mut net = compile_creature(&incumbent).map_err(|e| e.to_string())?;
        let report = accumulate_creature_learning_report(
            &incumbent,
            &mut net,
            req.training_data,
            req.config,
            req.max_records,
            &mut rng,
        )?;
        let candidate =
            apply_learnings_with(&incumbent, &report.learning, req.config, lr, req.apply);
        let deltas = count_apply_deltas(&incumbent, &candidate, req.config.plank_constant);
        let mut cand_net = compile_creature(&candidate).map_err(|e| e.to_string())?;
        let (after_mse, _) = compute_mse(
            &candidate,
            &mut cand_net,
            req.training_data,
            req.max_records,
        )?;
        fs::write(
            req.output_dir.join("candidate.json"),
            creature_to_json_pretty(&candidate).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let accepted = req.accept_always || after_mse < best_mse;
        if accepted {
            incumbent = candidate;
            best_mse = after_mse;
            accepted_epochs += 1;
        }
        let rec = TrainEpochRecord {
            kind: "epoch".into(),
            epoch,
            before_mse: report.mse,
            after_mse,
            records: report.records,
            accepted,
            hidden_biases: deltas.hidden_biases,
            output_biases: deltas.output_biases,
            hidden_weights: deltas.hidden_weights,
            output_weights: deltas.output_weights,
        };
        journal.push_str(&serde_json::to_string(&rec).map_err(|e| e.to_string())?);
        journal.push('\n');
        eprintln!(
            "epoch {epoch}: before_mse={:.12} after_mse={:.12} accepted={accepted} hidden_b={} out_b={} hidden_w={} out_w={}",
            report.mse,
            after_mse,
            deltas.hidden_biases,
            deltas.output_biases,
            deltas.hidden_weights,
            deltas.output_weights
        );
    }

    let best_json = creature_to_json_pretty(&incumbent).map_err(|e| e.to_string())?;
    fs::write(req.output_dir.join("best.json"), &best_json).map_err(|e| e.to_string())?;
    fs::write(&journal_path, journal).map_err(|e| e.to_string())?;

    let mut baseline_score = None;
    let mut best_score = None;
    if let Some(scorer) = req.scorer {
        let score_dir = req.output_dir.join("scorer-work");
        baseline_score = Some(score_creature(
            scorer,
            &text,
            req.training_data,
            &score_dir.join("baseline"),
        )?);
        best_score = Some(score_creature(
            scorer,
            &best_json,
            req.training_data,
            &score_dir.join("best"),
        )?);
    }

    Ok(TrainResult {
        creature: incumbent,
        baseline_mse,
        best_mse,
        accepted_epochs,
        baseline_score,
        best_score,
    })
}

/// Default output directory name.
pub fn default_output_dir() -> PathBuf {
    PathBuf::from(".backprop")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backprop::BackpropConfig;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn journal_header_carries_crate_version() {
        let header = TrainJournalHeader {
            kind: "runHeader".into(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            seed: 1,
            epochs: 1,
            max_records: Some(4),
            learning_rate: 0.01,
            step_scale: 1.0,
            outputs_only: false,
        };
        assert_eq!(header.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn one_epoch_identity_chain_can_accept() {
        let dir = tempdir().unwrap();
        let data = dir.path().join("data");
        fs::create_dir_all(&data).unwrap();
        let mut f = fs::File::create(data.join("0.bin")).unwrap();
        // input=1 → identity chain → 1; target=2 so there is error to learn.
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
        let result = run_train(TrainRequest {
            creature: &creature_path,
            training_data: &data,
            config: &cfg,
            epochs: 1,
            max_records: Some(1),
            seed: 1,
            output_dir: &out,
            scorer: None,
            apply: ApplyOptions::default(),
            accept_always: false,
        })
        .unwrap();
        assert!(result.baseline_mse > 0.0);
        assert!(out.join("best.json").is_file());
        assert!(out.join("journal.jsonl").is_file());
    }

    #[test]
    fn rejects_reentrant_creature() {
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
              "semanticVersion":"4.0.0","forwardOnly":false,"input":1,"output":1,
              "neurons":[{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
              "synapses":[{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
            }"#,
        )
        .unwrap();
        let err = run_train(TrainRequest {
            creature: &creature_path,
            training_data: &data,
            config: &BackpropConfig::default(),
            epochs: 1,
            max_records: Some(1),
            seed: 1,
            output_dir: &dir.path().join("out"),
            scorer: None,
            apply: ApplyOptions::default(),
            accept_always: false,
        })
        .unwrap_err();
        assert!(err.contains("forward-only"));
    }
}
