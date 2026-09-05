//! Evidence-driven sparse target selection, end to end (issue #108).
//!
//! Uniform random sparse selection spends a scorer run per target regardless
//! of whether the accumulation pass saw any usable signal there. These tests
//! assert the four properties the evidence strategy exists for: the ranking
//! follows the accumulated evidence rather than export order, the random
//! control arm is exactly the configured fraction, every candidate records
//! *why* its target was chosen, and a scored run reports wins/hour and score
//! gain/hour for both arms so the heuristic can be judged against uniform
//! random selection.
//!
//! The `rust_scorer` binary is a legitimate boundary to fake (see
//! `scorer_boundary.rs`): the stub prints the next score from a list.

#![cfg(unix)]

use neat_ai_backpropagation::backprop::{
    BackpropConfig, BiasSignal, LearningSignal, WeightSignal, calculate_learning_rate,
};
use neat_ai_backpropagation::blocks::{
    BlockGraph, BlockPlan, BlockStrategy, plan_blocks, proposal_magnitudes,
};
use neat_ai_backpropagation::blockwise::{BlocksRequest, BlocksSummary, run_blocks};
use neat_ai_backpropagation::propagate_layout::NeuronTraceStats;
use neat_ai_backpropagation::targets::{
    ArmSample, TargetPlan, TargetSource, TargetStrategy, compare_arms, rank_targets, select_targets,
};
use neat_ai_backpropagation::train::DEFAULT_MIN_SCORE_IMPROVEMENT;
use neat_core::{CreatureExport, parse_creature_json};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// Four hidden neurons feeding one output, so a target ranking has something
/// to order: `h_loud` carries the error mass and the proposal, `h_quiet` is
/// observed but flat, `h_dead` was never activated.
const CREATURE: &str = r#"{
  "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
  "neurons":[
    {"type":"hidden","uuid":"h_loud","bias":0.1,"squash":"IDENTITY"},
    {"type":"hidden","uuid":"h_quiet","bias":0.1,"squash":"IDENTITY"},
    {"type":"hidden","uuid":"h_dead","bias":0.1,"squash":"IDENTITY"},
    {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
  ],
  "synapses":[
    {"fromUUID":"input-0","toUUID":"h_loud","weight":1.0},
    {"fromUUID":"input-0","toUUID":"h_quiet","weight":1.0},
    {"fromUUID":"input-0","toUUID":"h_dead","weight":1.0},
    {"fromUUID":"h_loud","toUUID":"o1","weight":0.5},
    {"fromUUID":"h_quiet","toUUID":"o1","weight":0.5},
    {"fromUUID":"h_dead","toUUID":"o1","weight":0.5}
  ]
}"#;

/// Records in the generated corpus.
const RECORDS: u64 = 32;

fn creature() -> CreatureExport {
    parse_creature_json(CREATURE).expect("parse creature")
}

/// Index of `uuid` in the export neuron list.
fn neuron_index(creature: &CreatureExport, uuid: &str) -> usize {
    creature
        .neurons
        .iter()
        .position(|n| n.uuid == uuid)
        .unwrap_or_else(|| panic!("neuron {uuid} missing"))
}

/// A hand-built accumulation outcome: `h_loud` has error mass, a consistent
/// per-record direction and a proposal; `h_quiet` has a tiny, inconsistent
/// one; `h_dead` has nothing at all.
fn evidence(creature: &CreatureExport) -> (LearningSignal, Vec<NeuronTraceStats>) {
    let mut signal = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
    let mut traces = vec![NeuronTraceStats::default(); creature.neurons.len()];

    let loud = neuron_index(creature, "h_loud");
    signal.biases[loud] = BiasSignal {
        count: 32.0,
        total_adjusted_bias: 96.0,
        no_change: false,
    };
    traces[loud] = NeuronTraceStats {
        records: 32,
        total_bias: 12.0,
        total_error_absolute: 40.0,
        total_activation: 16.0,
        maximum_activation: 1.0,
        minimum_activation: -1.0,
        hint_value: 0.5,
    };

    let quiet = neuron_index(creature, "h_quiet");
    signal.biases[quiet] = BiasSignal {
        count: 32.0,
        total_adjusted_bias: 3.3,
        no_change: false,
    };
    traces[quiet] = NeuronTraceStats {
        records: 32,
        total_bias: 0.2,
        total_error_absolute: 0.5,
        total_activation: 0.1,
        maximum_activation: 0.02,
        minimum_activation: 0.0,
        hint_value: 0.0,
    };

    // Synapse 3 (h_loud → o1) pulls the same way on every record; synapse 4
    // (h_quiet → o1) is split, so its neuron reads as inconsistent.
    signal.weights[3] = WeightSignal {
        count: 32.0,
        total_positive_activation: 16.0,
        count_positive: 32.0,
        total_positive_adjusted_value: 48.0,
        ..WeightSignal::default()
    };
    signal.weights[4] = WeightSignal {
        count: 32.0,
        total_positive_activation: 0.5,
        total_negative_activation: -0.5,
        count_positive: 16.0,
        count_negative: 16.0,
        total_positive_adjusted_value: 0.6,
        total_negative_adjusted_value: -0.6,
    };
    (signal, traces)
}

/// The hidden neurons, in export order — the pool a focus strategy draws from.
fn hidden_pool(creature: &CreatureExport) -> Vec<usize> {
    creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type != "output")
        .map(|(i, _)| i)
        .collect()
}

/// Rank the hand-built evidence above.
fn ranked(creature: &CreatureExport) -> Vec<neat_ai_backpropagation::targets::RankedTarget> {
    let config = BackpropConfig::default();
    let learning_rate = calculate_learning_rate(&config, 0, None);
    let (signal, traces) = evidence(creature);
    let graph = BlockGraph::of(creature);
    let magnitudes = proposal_magnitudes(creature, &signal, &config, learning_rate, 1.0);
    rank_targets(
        creature,
        &graph,
        &magnitudes,
        &signal,
        &traces,
        &hidden_pool(creature),
    )
}

/// Criterion: candidates are ranked by the evidence one accumulation pass
/// produced — the neuron carrying the error mass and the proposal comes first,
/// the neuron the pass never activated comes last, whatever the export order.
#[test]
fn evidence_ranks_the_loud_neuron_first_and_the_unobserved_one_last() {
    let creature = creature();
    let ranked = ranked(&creature);
    assert_eq!(ranked.len(), 3, "one entry per eligible target");
    assert_eq!(ranked[0].neuron, neuron_index(&creature, "h_loud"));
    assert_eq!(ranked[0].rank, 0);
    assert_eq!(
        ranked[2].neuron,
        neuron_index(&creature, "h_dead"),
        "an unobserved neuron cannot outrank an observed one"
    );
    assert!(
        ranked[0].score > ranked[1].score && ranked[1].score >= ranked[2].score,
        "scores must be ordered: {:?}",
        ranked.iter().map(|t| t.score).collect::<Vec<_>>()
    );
    // Criterion: the rank features are recorded, not just the verdict.
    let loud = &ranked[0].features;
    assert_eq!(loud.activation_records, 32);
    assert!(loud.error_mass > 0.0, "error mass recorded");
    assert!(loud.relative_proposal > 0.0, "proposal magnitude recorded");
    assert!(
        loud.signal_consistency > ranked[1].features.signal_consistency,
        "a one-directional signal must read as more consistent than a split one"
    );
    assert_eq!(loud.fan_in, 1);
    assert_eq!(loud.fan_out, 1);
    assert!(loud.activation_range > 0.0, "activity range recorded");
    let dead = &ranked[2].features;
    assert_eq!(dead.activation_records, 0);
    assert_eq!(dead.error_mass, 0.0);
}

/// Criterion: a configurable exploitation vs random-control ratio. Half of a
/// four-target draw is the control arm, and the other half is the top of the
/// ranking — the control targets are drawn from what exploitation left.
#[test]
fn the_random_control_fraction_splits_the_draw() {
    let creature = creature();
    let ranked = ranked(&creature);
    let plan = TargetPlan {
        strategy: TargetStrategy::Evidence,
        random_control_fraction: 0.5,
    };
    let mut rng = StdRng::seed_from_u64(7);
    let selected = select_targets(&plan, &ranked, 2, &mut rng);
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].selection.source, TargetSource::Evidence);
    assert_eq!(selected[0].selection.rank, 0);
    assert_eq!(selected[0].neuron, neuron_index(&creature, "h_loud"));
    assert_eq!(selected[1].selection.source, TargetSource::RandomControl);
    assert_ne!(
        selected[1].neuron, selected[0].neuron,
        "the control arm must not re-draw the exploited target"
    );
    // A control target still carries the features it was ranked on, so a win
    // found by accident is as explainable as an exploited one.
    assert!(selected[1].selection.rank > 0);

    // Zero control fraction is pure exploitation, in rank order.
    let exploit_only = TargetPlan {
        strategy: TargetStrategy::Evidence,
        random_control_fraction: 0.0,
    };
    let selected = select_targets(&exploit_only, &ranked, 3, &mut rng);
    assert_eq!(
        selected
            .iter()
            .map(|t| t.selection.rank)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert!(
        selected
            .iter()
            .all(|t| t.selection.source == TargetSource::Evidence)
    );
}

/// Criterion: the existing uniform random strategy is retained — it draws every
/// target uniformly and labels the whole draw as the control arm.
#[test]
fn the_random_strategy_draws_every_target_uniformly() {
    let creature = creature();
    let ranked = ranked(&creature);
    let plan = TargetPlan {
        strategy: TargetStrategy::Random,
        random_control_fraction: 0.0,
    };
    let mut counts = [0usize; 3];
    let mut first_ranks = Vec::new();
    for seed in 0..64u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let selected = select_targets(&plan, &ranked, 1, &mut rng);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].selection.source, TargetSource::RandomControl);
        counts[selected[0].neuron] += 1;
        first_ranks.push(selected[0].selection.rank);
    }
    assert!(
        counts.iter().all(|&c| c > 0),
        "a uniform draw must reach every eligible target: {counts:?}"
    );
    assert!(
        first_ranks.iter().any(|&r| r > 0),
        "uniform selection must not collapse onto the top-ranked target"
    );
}

/// A plan that could never do what it claims is refused rather than silently
/// clamped — a fraction above one would read as "a control arm ran".
#[test]
fn an_impossible_control_fraction_is_refused() {
    for fraction in [-0.1, 1.5, f64::NAN] {
        let plan = BlockPlan {
            targets: TargetPlan {
                strategy: TargetStrategy::Evidence,
                random_control_fraction: fraction,
            },
            ..BlockPlan::default()
        };
        let err = plan
            .validate()
            .expect_err("fraction {fraction} must be refused");
        assert!(
            err.contains("randomControlFraction"),
            "error must name the setting: {err}"
        );
    }
}

/// Criterion: ranked target / neighbourhood candidates come out of one
/// accumulation pass, and each block records why its target was selected.
#[test]
fn planned_blocks_carry_their_selection_evidence() {
    let creature = creature();
    let config = BackpropConfig::default();
    let learning_rate = calculate_learning_rate(&config, 0, None);
    let (signal, traces) = evidence(&creature);
    let graph = BlockGraph::of(&creature);
    let magnitudes = proposal_magnitudes(&creature, &signal, &config, learning_rate, 1.0);
    let plan = BlockPlan {
        strategies: vec![BlockStrategy::Neuron, BlockStrategy::Neighbourhood],
        blocks_per_strategy: 1,
        targets: TargetPlan {
            strategy: TargetStrategy::Evidence,
            random_control_fraction: 0.0,
        },
        ..BlockPlan::default()
    };
    let mut rng = StdRng::seed_from_u64(5);
    let planned = plan_blocks(
        &creature,
        &graph,
        &magnitudes,
        &signal,
        &traces,
        &plan,
        &mut rng,
    )
    .expect("plan blocks");
    let loud = neuron_index(&creature, "h_loud");
    for block in &planned.blocks {
        assert_eq!(
            block.focus,
            Some(loud),
            "evidence selection must focus the loud neuron"
        );
        let selection = block
            .selection
            .as_ref()
            .expect("a focus block records its selection");
        assert_eq!(selection.source, TargetSource::Evidence);
        assert_eq!(selection.rank, 0);
        assert!(selection.features.error_mass > 0.0);
    }
}

/// Criterion: scorer wins/hour and score gain/hour, per arm, so the heuristic
/// can be compared against uniform random selection.
#[test]
fn arm_throughput_reports_wins_and_gain_per_hour() {
    let samples = vec![
        ArmSample {
            source: TargetSource::Evidence,
            score_delta: Some(2.0),
            win: true,
            scorer_seconds: 900.0,
        },
        ArmSample {
            source: TargetSource::Evidence,
            score_delta: Some(-1.0),
            win: false,
            scorer_seconds: 900.0,
        },
        ArmSample {
            source: TargetSource::RandomControl,
            score_delta: Some(0.5),
            win: true,
            scorer_seconds: 3600.0,
        },
    ];
    let comparison = compare_arms(&samples);
    assert_eq!(comparison.evidence.candidates_scored, 2);
    assert_eq!(comparison.evidence.wins, 1);
    // One win in half an hour of scorer time.
    assert_eq!(comparison.evidence.wins_per_hour, Some(2.0));
    // A loss is not a negative gain in the rate — a rejected candidate is
    // rolled back, so only the gains that survived count.
    assert_eq!(comparison.evidence.total_score_gain, 2.0);
    assert_eq!(comparison.evidence.score_gain_per_hour, Some(4.0));
    assert_eq!(comparison.evidence.best_score_delta, Some(2.0));
    assert_eq!(comparison.random_control.wins_per_hour, Some(1.0));
    assert_eq!(comparison.random_control.score_gain_per_hour, Some(0.5));

    // No scorer time is no rate at all — never a division that reads as an
    // infinite win rate.
    let idle = compare_arms(&[ArmSample {
        source: TargetSource::Evidence,
        score_delta: None,
        win: false,
        scorer_seconds: 0.0,
    }]);
    assert_eq!(idle.evidence.wins_per_hour, None);
    assert_eq!(idle.evidence.score_gain_per_hour, None);
    assert_eq!(idle.random_control.candidates_scored, 0);
}

/// End to end: a scored blocks run records the selection metadata on every
/// candidate and compares the two arms in `blocks.json`.
#[test]
fn a_scored_run_compares_the_evidence_and_control_arms() {
    let fixture = Fixture::new(&["0.1", "0.9", "0.2", "0.9", "0.2"]);
    let summary = fixture.run(
        "out",
        &BlockPlan {
            strategies: vec![BlockStrategy::Neuron],
            blocks_per_strategy: 2,
            targets: TargetPlan {
                strategy: TargetStrategy::Evidence,
                random_control_fraction: 0.5,
            },
            ..BlockPlan::default()
        },
    );
    let sources: Vec<TargetSource> = summary
        .candidates
        .iter()
        .filter_map(|c| c.selection.as_ref().map(|s| s.source))
        .collect();
    assert_eq!(
        sources.len(),
        2,
        "every focus candidate records its selection"
    );
    assert!(sources.contains(&TargetSource::Evidence));
    assert!(sources.contains(&TargetSource::RandomControl));

    let comparison = summary
        .selection_comparison
        .as_ref()
        .expect("a scored run compares the arms");
    assert_eq!(comparison.evidence.candidates_scored, 1);
    assert_eq!(comparison.random_control.candidates_scored, 1);
    assert!(
        comparison.evidence.scorer_seconds > 0.0,
        "scorer time must be measured, not assumed"
    );
    assert!(
        comparison.evidence.wins_per_hour.is_some(),
        "measured scorer time yields a rate"
    );

    // The same numbers must survive the round trip to disk.
    let json =
        fs::read_to_string(fixture.root.join("out").join("blocks.json")).expect("read blocks.json");
    let reread: BlocksSummary = serde_json::from_str(&json).expect("parse blocks.json");
    assert_eq!(
        reread
            .selection_comparison
            .expect("comparison round-trips")
            .evidence
            .wins,
        comparison.evidence.wins
    );
    assert!(
        json.contains("randomControl"),
        "the candidate metadata names the control arm"
    );
}

struct Fixture {
    _dir: TempDir,
    root: PathBuf,
    creature: PathBuf,
    data: PathBuf,
    scorer: PathBuf,
}

impl Fixture {
    /// Creature, corpus and a stub scorer handing back `scores` in call order
    /// (the last value repeats).
    fn new(scores: &[&str]) -> Self {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let data = root.join("data");
        fs::create_dir_all(&data).expect("create data dir");
        let mut f = fs::File::create(data.join("0.bin")).expect("create records");
        for i in 0..RECORDS {
            let x = i as f32 / (RECORDS as f32 / 2.0) - 1.0;
            f.write_all(&x.to_le_bytes()).expect("write input");
            f.write_all(&(0.5 * x + 0.25).to_le_bytes())
                .expect("write target");
        }
        let creature_path = root.join("creature.json");
        fs::write(&creature_path, CREATURE).expect("write creature");
        let scores_file = root.join("scores.txt");
        fs::write(&scores_file, format!("{}\n", scores.join("\n"))).expect("write scores");
        let scorer = write_stub_scorer(&root, &scores_file, &root.join("calls.txt"));
        Self {
            _dir: dir,
            root,
            creature: creature_path,
            data,
            scorer,
        }
    }

    fn run(&self, out_name: &str, plan: &BlockPlan) -> BlocksSummary {
        run_blocks(BlocksRequest {
            creature: &self.creature,
            training_data: &self.data,
            config: &BackpropConfig::default(),
            max_records: None,
            seed: 3,
            step_scale: 0.5,
            plan,
            skip_mse: true,
            scorer: Some(self.scorer.as_path()),
            min_score_improvement: DEFAULT_MIN_SCORE_IMPROVEMENT,
            output_dir: &self.root.join(out_name),
        })
        .expect("blockwise run")
    }
}

/// Write an executable `/bin/sh` stub standing in for `rust_scorer`.
fn write_stub_scorer(dir: &Path, scores: &Path, calls: &Path) -> PathBuf {
    let path = dir.join("stub-scorer");
    let body = format!(
        r#"#!/bin/sh
set -eu
printf 'call\n' >> '{calls}'
n=$(wc -l < '{calls}' | tr -d ' ')
score=$(sed -n "${{n}}p" '{scores}')
if [ -z "$score" ]; then
  score=$(tail -n 1 '{scores}')
fi
printf '{{"trained":{{"score":%s,"error":0.0}}}}\n' "$score"
"#,
        calls = calls.display(),
        scores = scores.display(),
    );
    fs::write(&path, body).expect("write stub scorer");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod stub scorer");
    path
}
