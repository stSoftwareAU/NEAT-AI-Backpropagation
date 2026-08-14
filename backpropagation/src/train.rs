//! `trainDir`-style epoch loop: accumulate → apply if MSE improved → else rollback.

use crate::backprop::{
    ApplyOptions, BackpropConfig, apply_learnings_with, calculate_learning_rate, count_apply_deltas,
};
use crate::mse::compute_mse;
use crate::propagate_layout::accumulate_creature_learning_report;
use crate::scorer::{ScoreResult, score_creature};
use crate::tags::{BackpropProgress, CreatureMeta, serialize_creature_with_meta};
use neat_core::{CreatureExport, compile_creature, creature_to_json_pretty, parse_creature_json};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Default apply step scale for a `train` run (#39).
///
/// Every gene's proposal is computed as if the others stay put, so moving all
/// of them the whole way at once overshoots on large creatures. `sweep`'s
/// default grid tops out at 1% — the trainer starts there rather than 100×
/// above it, and the backtracking line search shrinks further when needed.
pub const DEFAULT_STEP_SCALE: f64 = 0.01;

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
    /// Learning rate this epoch resolved from the configured strategy (#39).
    #[serde(default)]
    pub learning_rate: f64,
    /// Whether the candidate was kept.
    pub accepted: bool,
    /// Step-scale halvings tried after the initial step this epoch (#38).
    #[serde(default)]
    pub backtracks: u32,
    /// Step scale of the final (kept or last-tried) candidate (#38).
    #[serde(default)]
    pub step_scale: f64,
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
    /// Backtracking line search (#38): on a rejected apply, retry the same
    /// accumulated learning at step/2, step/4, … up to this many halvings
    /// before declaring the epoch dry. `0` = single attempt (old behaviour).
    pub max_backtracks: u32,
}

/// Run the experimental trainer and write `journal.jsonl` + `best.json`.
pub fn run_train(req: TrainRequest<'_>) -> Result<TrainResult, String> {
    fs::create_dir_all(req.output_dir).map_err(|e| e.to_string())?;
    let text = fs::read_to_string(req.creature).map_err(|e| e.to_string())?;
    let mut meta = CreatureMeta::from_creature_json(&text);
    let mut incumbent = parse_creature_json(&text).map_err(|e| e.to_string())?;
    if !incumbent.forward_only {
        return Err(
            "this trainer supports forward-only creatures only (no re-entrant / recurrent graphs)"
                .into(),
        );
    }
    let initial_lr = calculate_learning_rate(req.config, 0, None);

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
        learning_rate: initial_lr,
        step_scale: req.apply.step_scale,
        outputs_only: req.apply.outputs_only,
    };
    let mut journal = String::new();
    journal.push_str(&serde_json::to_string(&header).map_err(|e| e.to_string())?);
    journal.push('\n');

    let mut best_mse = baseline_mse;
    let mut accepted_epochs = 0u64;
    let mut rng = StdRng::seed_from_u64(req.seed);
    let mut previous_mse: Option<f64> = None;

    for epoch in 1..=req.epochs {
        // Resolve the epoch's rate from the configured strategy (#39): fixed
        // holds steady, decay / warm-restart follow the epoch index, adaptive
        // reacts to the last epoch's MSE movement.
        let lr = calculate_learning_rate(
            req.config,
            epoch - 1,
            previous_mse.map(|prev| (prev, best_mse)),
        );
        previous_mse = Some(best_mse);
        let mut net = compile_creature(&incumbent).map_err(|e| e.to_string())?;
        let report = accumulate_creature_learning_report(
            &incumbent,
            &mut net,
            req.training_data,
            req.config,
            req.max_records,
            &mut rng,
        )?;
        // Backtracking line search (#38): the accumulate above is the
        // expensive part — on a rejected apply, halve the step and re-test
        // the same learning instead of discarding the epoch.
        let mut step_scale = if req.apply.step_scale.is_finite() && req.apply.step_scale > 0.0 {
            req.apply.step_scale.min(1.0)
        } else {
            1.0
        };
        let mut backtracks = 0u32;
        let (candidate, deltas, after_mse, accepted) = loop {
            let candidate = apply_learnings_with(
                &incumbent,
                &report.learning,
                req.config,
                lr,
                ApplyOptions {
                    step_scale,
                    ..req.apply
                },
            );
            let deltas = count_apply_deltas(&incumbent, &candidate, req.config.plank_constant);
            let mut cand_net = compile_creature(&candidate).map_err(|e| e.to_string())?;
            let (after_mse, _) = compute_mse(
                &candidate,
                &mut cand_net,
                req.training_data,
                req.max_records,
            )?;
            let accepted = req.accept_always || after_mse < best_mse;
            if accepted || backtracks >= req.max_backtracks {
                break (candidate, deltas, after_mse, accepted);
            }
            backtracks += 1;
            step_scale /= 2.0;
        };
        fs::write(
            req.output_dir.join("candidate.json"),
            creature_to_json_pretty(&candidate).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
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
            learning_rate: lr,
            accepted,
            backtracks,
            step_scale,
            hidden_biases: deltas.hidden_biases,
            output_biases: deltas.output_biases,
            hidden_weights: deltas.hidden_weights,
            output_weights: deltas.output_weights,
        };
        journal.push_str(&serde_json::to_string(&rec).map_err(|e| e.to_string())?);
        journal.push('\n');
        eprintln!(
            "epoch {epoch}: before_mse={:.12} after_mse={:.12} accepted={accepted} backtracks={backtracks} step={step_scale:.8} hidden_b={} out_b={} hidden_w={} out_w={}",
            report.mse,
            after_mse,
            deltas.hidden_biases,
            deltas.output_biases,
            deltas.hidden_weights,
            deltas.output_weights
        );
        // With deterministic accumulation (full sparse ratio, no random
        // samples) a rejected epoch would recompute the identical learning —
        // further epochs cannot make progress, so stop early (#38).
        if !accepted && req.config.sparse_ratio >= 1.0 && req.config.disable_random_samples {
            eprintln!(
                "epoch {epoch}: rejected after {backtracks} backtracks and accumulation is deterministic; stopping early (#38)"
            );
            break;
        }
    }

    // Score before writing best.json so GRQ can read `score` / `backpropagation`
    // tags without a second rescore pass (GRQ #3991). Untagged best.json when
    // --scorer is omitted — callers that need the gate must pass a scorer.
    let mut baseline_score = None;
    let mut best_score = None;
    let compact_best = creature_to_json_pretty(&incumbent).map_err(|e| e.to_string())?;
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
            &compact_best,
            req.training_data,
            &score_dir.join("best"),
        )?);
        if let (Some(baseline), Some(best)) = (&baseline_score, &best_score) {
            meta.stamp_train_result(&BackpropProgress {
                accepted_epochs,
                epochs: req.epochs,
                score: best.score,
                error: best.error,
                opening_score: baseline.score,
            });
        }
    }

    let best_json = serialize_creature_with_meta(&incumbent, &meta)?;
    fs::write(req.output_dir.join("best.json"), &best_json).map_err(|e| e.to_string())?;
    fs::write(&journal_path, journal).map_err(|e| e.to_string())?;

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

    /// Journalling the crate version is what remote GRQ runners use to spot a
    /// stale binary, so assert it off a real `run_train` journal rather than
    /// off a struct literal (#25).
    #[test]
    fn journal_header_carries_crate_version() {
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
        run_train(TrainRequest {
            creature: &creature_path,
            training_data: &data,
            config: &BackpropConfig::default(),
            epochs: 1,
            max_records: Some(1),
            seed: 7,
            output_dir: &out,
            scorer: None,
            apply: ApplyOptions::default(),
            accept_always: false,
            max_backtracks: 0,
        })
        .unwrap();

        let journal = fs::read_to_string(out.join("journal.jsonl")).unwrap();
        let first_line = journal.lines().next().expect("journal has a header line");
        let header: TrainJournalHeader = serde_json::from_str(first_line).unwrap();
        assert_eq!(header.kind, "runHeader");
        assert_eq!(header.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(header.seed, 7);
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
            max_backtracks: 0,
        })
        .unwrap();
        assert!(result.baseline_mse > 0.0);
        assert!(out.join("best.json").is_file());
        assert!(out.join("journal.jsonl").is_file());
    }

    #[test]
    fn rejected_deterministic_epoch_stops_early() {
        let dir = tempdir().unwrap();
        let data = dir.path().join("data");
        fs::create_dir_all(&data).unwrap();
        let mut f = fs::File::create(data.join("0.bin")).unwrap();
        // input=1 → identity chain → 1; target=1 so MSE is already 0 and no
        // apply can strictly improve it — every epoch must reject.
        f.write_all(&1.0f32.to_le_bytes()).unwrap();
        f.write_all(&1.0f32.to_le_bytes()).unwrap();
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
        let result = run_train(TrainRequest {
            creature: &creature_path,
            training_data: &data,
            config: &BackpropConfig::default(),
            epochs: 5,
            max_records: Some(1),
            seed: 1,
            output_dir: &out,
            scorer: None,
            apply: ApplyOptions::default(),
            accept_always: false,
            max_backtracks: 2,
        })
        .unwrap();
        assert_eq!(result.accepted_epochs, 0);
        let journal = fs::read_to_string(out.join("journal.jsonl")).unwrap();
        let epoch_lines = journal
            .lines()
            .filter(|l| l.contains("\"kind\":\"epoch\""))
            .count();
        // Deterministic accumulation + rejection → early stop after epoch 1,
        // not 5 identical rejected epochs (#38).
        assert_eq!(epoch_lines, 1);
    }

    #[test]
    fn backtracking_accepts_when_full_step_overshoots() {
        let dir = tempdir().unwrap();
        let data = dir.path().join("data");
        fs::create_dir_all(&data).unwrap();
        let mut f = fs::File::create(data.join("0.bin")).unwrap();
        // input=1 → identity chain → 1; target=1.5. With a large learning
        // rate the proposed jump overshoots past the target (worse MSE at
        // full step) but a halved step lands closer — the line search must
        // find it instead of rejecting the epoch.
        f.write_all(&1.0f32.to_le_bytes()).unwrap();
        f.write_all(&1.5f32.to_le_bytes()).unwrap();
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
        let cfg = BackpropConfig {
            learning_rate: 1.0,
            initial_learning_rate: 1.0,
            ..BackpropConfig::default()
        };
        let out_no_ls = dir.path().join("out-no-ls");
        let baseline = run_train(TrainRequest {
            creature: &creature_path,
            training_data: &data,
            config: &cfg,
            epochs: 1,
            max_records: Some(1),
            seed: 1,
            output_dir: &out_no_ls,
            scorer: None,
            apply: ApplyOptions::default(),
            accept_always: false,
            max_backtracks: 0,
        })
        .unwrap();
        let out_ls = dir.path().join("out-ls");
        let with_ls = run_train(TrainRequest {
            creature: &creature_path,
            training_data: &data,
            config: &cfg,
            epochs: 1,
            max_records: Some(1),
            seed: 1,
            output_dir: &out_ls,
            scorer: None,
            apply: ApplyOptions::default(),
            accept_always: false,
            max_backtracks: 8,
        })
        .unwrap();
        // The line-search run must never do worse than the single-attempt
        // run, and when the full step overshoots it must recover an accept.
        assert!(with_ls.best_mse <= baseline.best_mse);
        if baseline.accepted_epochs == 0 {
            assert_eq!(with_ls.accepted_epochs, 1);
            let journal = fs::read_to_string(out_ls.join("journal.jsonl")).unwrap();
            assert!(journal.contains("\"backtracks\":"));
        }
    }

    /// Two dense identity layers into one output — many genes, many paths, so
    /// per-gene proposals (each computed as if the others hold still) compound
    /// when applied together.
    fn dense_creature_json(layer_a: usize, layer_b: usize) -> String {
        let mut neurons = Vec::new();
        for i in 0..layer_a {
            neurons.push(format!(
                r#"{{"type":"hidden","uuid":"a{i}","bias":0.01,"squash":"IDENTITY"}}"#
            ));
        }
        for j in 0..layer_b {
            neurons.push(format!(
                r#"{{"type":"hidden","uuid":"b{j}","bias":0.01,"squash":"IDENTITY"}}"#
            ));
        }
        neurons.push(r#"{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}"#.to_string());
        let mut synapses = Vec::new();
        for i in 0..layer_a {
            synapses.push(format!(
                r#"{{"fromUUID":"input-0","toUUID":"a{i}","weight":0.1}}"#
            ));
            for j in 0..layer_b {
                synapses.push(format!(
                    r#"{{"fromUUID":"a{i}","toUUID":"b{j}","weight":0.05}}"#
                ));
            }
        }
        for j in 0..layer_b {
            synapses.push(format!(
                r#"{{"fromUUID":"b{j}","toUUID":"o1","weight":0.05}}"#
            ));
        }
        format!(
            r#"{{"semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
              "neurons":[{}],"synapses":[{}]}}"#,
            neurons.join(","),
            synapses.join(",")
        )
    }

    /// `y = 0.5x + 0.25` over a deterministic sweep of inputs.
    fn write_linear_records(path: &Path, count: usize) {
        let mut f = fs::File::create(path).unwrap();
        for i in 0..count {
            let x = (i as f32) / (count as f32) * 2.0 - 1.0;
            f.write_all(&x.to_le_bytes()).unwrap();
            f.write_all(&(0.5 * x + 0.25).to_le_bytes()).unwrap();
        }
    }

    fn run_at_step(
        creature: &Path,
        data: &Path,
        out: &Path,
        step_scale: f64,
        learning_rate: f64,
    ) -> TrainResult {
        let cfg = BackpropConfig {
            learning_rate,
            initial_learning_rate: learning_rate,
            ..BackpropConfig::default()
        };
        run_train(TrainRequest {
            creature,
            training_data: data,
            config: &cfg,
            epochs: 1,
            max_records: None,
            seed: 1,
            output_dir: out,
            scorer: None,
            apply: ApplyOptions {
                step_scale,
                ..ApplyOptions::default()
            },
            // Keep the candidate either way so the raw post-apply MSE of each
            // step scale is comparable.
            accept_always: true,
            max_backtracks: 0,
        })
        .unwrap()
    }

    #[test]
    fn default_step_scale_improves_where_the_full_step_overshoots() {
        let dir = tempdir().unwrap();
        let data = dir.path().join("data");
        fs::create_dir_all(&data).unwrap();
        write_linear_records(&data.join("0.bin"), 64);
        let creature_path = dir.path().join("creature.json");
        fs::write(&creature_path, dense_creature_json(8, 8)).unwrap();

        let full = run_at_step(
            &creature_path,
            &data,
            &dir.path().join("out-full"),
            1.0,
            0.5,
        );
        let default = run_at_step(
            &creature_path,
            &data,
            &dir.path().join("out-default"),
            DEFAULT_STEP_SCALE,
            0.5,
        );
        assert!(
            full.best_mse > full.baseline_mse,
            "full step should overshoot: {} -> {}",
            full.baseline_mse,
            full.best_mse
        );
        assert!(
            default.best_mse < default.baseline_mse,
            "sweep-informed step should improve: {} -> {}",
            default.baseline_mse,
            default.best_mse
        );
    }

    #[test]
    fn decay_strategy_lowers_the_learning_rate_each_epoch() {
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
        let cfg = BackpropConfig {
            learning_rate_strategy: crate::backprop::LearningRateStrategy::Decay,
            learning_rate: 0.1,
            initial_learning_rate: 0.1,
            learning_rate_decay: 0.5,
            ..BackpropConfig::default()
        };
        let out = dir.path().join("out");
        run_train(TrainRequest {
            creature: &creature_path,
            training_data: &data,
            config: &cfg,
            epochs: 3,
            max_records: Some(1),
            seed: 1,
            output_dir: &out,
            scorer: None,
            // accept_always keeps all three epochs running so the schedule is
            // observable end to end.
            apply: ApplyOptions::default(),
            accept_always: true,
            max_backtracks: 0,
        })
        .unwrap();
        let journal = fs::read_to_string(out.join("journal.jsonl")).unwrap();
        let rates: Vec<f64> = journal
            .lines()
            .filter(|l| l.contains("\"kind\":\"epoch\""))
            .map(|l| {
                serde_json::from_str::<TrainEpochRecord>(l)
                    .unwrap()
                    .learning_rate
            })
            .collect();
        assert_eq!(rates.len(), 3);
        assert!((rates[0] - 0.1).abs() < 1e-12, "epoch 1 lr {}", rates[0]);
        assert!((rates[1] - 0.05).abs() < 1e-12, "epoch 2 lr {}", rates[1]);
        assert!((rates[2] - 0.025).abs() < 1e-12, "epoch 3 lr {}", rates[2]);
    }

    #[test]
    fn fixed_strategy_journals_a_constant_learning_rate() {
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
              "neurons":[{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
              "synapses":[{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
            }"#,
        )
        .unwrap();
        let out = dir.path().join("out");
        run_train(TrainRequest {
            creature: &creature_path,
            training_data: &data,
            config: &BackpropConfig::default(),
            epochs: 2,
            max_records: Some(1),
            seed: 1,
            output_dir: &out,
            scorer: None,
            apply: ApplyOptions::default(),
            accept_always: true,
            max_backtracks: 0,
        })
        .unwrap();
        let journal = fs::read_to_string(out.join("journal.jsonl")).unwrap();
        for line in journal.lines().filter(|l| l.contains("\"kind\":\"epoch\"")) {
            let rec: TrainEpochRecord = serde_json::from_str(line).unwrap();
            assert!((rec.learning_rate - 0.01).abs() < 1e-12);
        }
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
            max_backtracks: 0,
        })
        .unwrap_err();
        assert!(err.contains("forward-only"));
    }
}
