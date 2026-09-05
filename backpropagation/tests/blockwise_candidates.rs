//! Blockwise candidate generation, end to end (issue #105).
//!
//! The production creature is large enough that one accumulated proposal moves
//! thousands of genes at once, and correlated moves destroy the compensating
//! relationships evolution found. `run_blocks` carves one accumulation pass
//! into small regions instead, so these tests assert the three properties the
//! mode exists for: the corpus is accumulated **once** however many blocks are
//! generated, every candidate is scored **independently**, and the written
//! metadata names exactly the genes the block was allowed to move.
//!
//! The `rust_scorer` binary is a legitimate boundary to fake (see
//! `scorer_boundary.rs`): the stub prints the next score from a list and
//! records that it was called, so both the verdicts and the number of
//! invocations are observable.

#![cfg(unix)]

use neat_ai_backpropagation::backprop::BackpropConfig;
use neat_ai_backpropagation::blocks::{BlockPlan, BlockStrategy};
use neat_ai_backpropagation::blockwise::{BlocksRequest, BlocksSummary, run_blocks};
use neat_ai_backpropagation::creature_io::load_forward_only_creature;
use neat_ai_backpropagation::train::DEFAULT_MIN_SCORE_IMPROVEMENT;
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// `input-0 → h1 → h2 → o1` with an `h1 → o1` shortcut: three movable neurons
/// and four synapses, so a neuron block, the output head and the whole
/// creature are genuinely different gene sets.
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

/// Records in the generated corpus.
const RECORDS: u64 = 32;

struct Fixture {
    _dir: TempDir,
    root: PathBuf,
    creature: PathBuf,
    data: PathBuf,
    scorer: PathBuf,
    calls: PathBuf,
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
        let creature = root.join("creature.json");
        fs::write(&creature, CHAIN).expect("write creature");

        let scores_file = root.join("scores.txt");
        fs::write(&scores_file, format!("{}\n", scores.join("\n"))).expect("write scores");
        let calls = root.join("calls.txt");
        let scorer = write_stub_scorer(&root, &scores_file, &calls);
        Self {
            _dir: dir,
            root,
            creature,
            data,
            scorer,
            calls,
        }
    }

    /// How many times the stub scorer was invoked so far.
    ///
    /// A missing file is zero calls; any other read error is a broken fixture
    /// and must not be reported as "the scorer never ran".
    fn scorer_calls(&self) -> usize {
        match fs::read_to_string(&self.calls) {
            Ok(text) => text.lines().count(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
            Err(e) => panic!("cannot read {}: {e}", self.calls.display()),
        }
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

/// Run blockwise generation over `fixture` with the given plan.
fn blocks(
    fixture: &Fixture,
    out_name: &str,
    plan: &BlockPlan,
    with_scorer: bool,
    skip_mse: bool,
) -> BlocksSummary {
    blocks_with(
        fixture,
        out_name,
        plan,
        with_scorer,
        skip_mse,
        &BackpropConfig::default(),
    )
}

/// [`blocks`] with an explicit backprop config.
fn blocks_with(
    fixture: &Fixture,
    out_name: &str,
    plan: &BlockPlan,
    with_scorer: bool,
    skip_mse: bool,
    config: &BackpropConfig,
) -> BlocksSummary {
    let out = fixture.root.join(out_name);
    run_blocks(BlocksRequest {
        creature: &fixture.creature,
        training_data: &fixture.data,
        config,
        max_records: None,
        seed: 3,
        step_scale: 0.5,
        plan,
        skip_mse,
        scorer: with_scorer.then_some(fixture.scorer.as_path()),
        min_score_improvement: DEFAULT_MIN_SCORE_IMPROVEMENT,
        output_dir: &out,
    })
    .expect("blockwise run")
}

/// Criterion: one accumulation pass can produce many candidate blocks without
/// rereading the corpus. A run generating every strategy consumes exactly the
/// records a single-block run consumes — one pass, however many candidates.
#[test]
fn one_accumulation_pass_serves_every_block() {
    let fixture = Fixture::new(&["0.5"]);
    let single = blocks(
        &fixture,
        "out-single",
        &BlockPlan {
            strategies: vec![BlockStrategy::Global],
            ..BlockPlan::default()
        },
        false,
        true,
    );
    let many = blocks(
        &fixture,
        "out-many",
        &BlockPlan {
            blocks_per_strategy: 3,
            subgraph_size: 2,
            top_genes: 2,
            ..BlockPlan::default()
        },
        false,
        true,
    );
    assert_eq!(single.candidates.len(), 1);
    assert!(
        many.candidates.len() > 4,
        "expected many blocks, got {}",
        many.candidates.len()
    );
    assert_eq!(single.records, RECORDS);
    assert_eq!(
        many.records,
        single.records,
        "generating {} candidates must not cost extra accumulation passes",
        many.candidates.len()
    );
}

/// The discriminating half of the property above: every block candidate must be
/// the *same* signal restricted to its genes.
///
/// A sparse ratio below 1 makes each accumulation pass draw its own random
/// neuron subset, so an implementation that re-accumulated per block would hand
/// different proposals to each candidate. Sharing one pass, the block genes
/// carry exactly the values the whole-creature candidate gave them.
#[test]
fn every_block_carves_the_same_accumulated_signal() {
    let fixture = Fixture::new(&["0.5"]);
    let config = BackpropConfig {
        sparse_ratio: 0.5,
        ..BackpropConfig::default()
    };
    let summary = blocks_with(
        &fixture,
        "out",
        &BlockPlan {
            strategies: vec![
                BlockStrategy::Global,
                BlockStrategy::Neuron,
                BlockStrategy::Neighbourhood,
                BlockStrategy::OutputHead,
            ],
            blocks_per_strategy: 2,
            ..BlockPlan::default()
        },
        false,
        true,
        &config,
    );
    let out = fixture.root.join("out");
    let global_record = summary
        .candidates
        .iter()
        .find(|c| c.strategy == BlockStrategy::Global)
        .expect("global block present");
    let global = load_forward_only_creature(
        &out.join(global_record.candidate.as_ref().expect("global written")),
    )
    .expect("global creature");
    let source = load_forward_only_creature(&fixture.creature).expect("source creature");

    let mut compared = 0usize;
    for candidate in summary
        .candidates
        .iter()
        .filter(|c| c.strategy != BlockStrategy::Global)
    {
        let Some(relative) = &candidate.candidate else {
            continue;
        };
        let applied = load_forward_only_creature(&out.join(relative)).expect("candidate creature");
        for (i, neuron) in applied.neurons.iter().enumerate() {
            let expected = if candidate.neurons.contains(&neuron.uuid) {
                global.neurons[i].bias
            } else {
                source.neurons[i].bias
            };
            assert_eq!(
                neuron.bias, expected,
                "{} bias of {}",
                candidate.label, neuron.uuid
            );
            compared += 1;
        }
        for (i, synapse) in applied.synapses.iter().enumerate() {
            let in_block = candidate
                .synapses
                .iter()
                .any(|s| s.from_uuid == synapse.from_uuid && s.to_uuid == synapse.to_uuid);
            let expected = if in_block {
                global.synapses[i].weight
            } else {
                source.synapses[i].weight
            };
            assert_eq!(
                synapse.weight, expected,
                "{} weight of {}→{}",
                candidate.label, synapse.from_uuid, synapse.to_uuid
            );
            compared += 1;
        }
    }
    assert!(compared > 0, "expected block candidates to compare against");
}

/// Criterion: each candidate can be scored independently, and the scorer — not
/// MSE — decides which block was useful.
#[test]
fn every_candidate_is_scored_on_its_own() {
    // Baseline 0.5, then one loss and wins thereafter.
    let fixture = Fixture::new(&["0.5", "0.4", "0.6"]);
    let summary = blocks(
        &fixture,
        "out",
        &BlockPlan {
            blocks_per_strategy: 2,
            subgraph_size: 2,
            top_genes: 2,
            ..BlockPlan::default()
        },
        true,
        true,
    );
    let written = summary.written();
    assert!(written.len() > 2, "expected several written candidates");
    // One baseline score plus one per written candidate — no candidate shares
    // another's verdict.
    assert_eq!(fixture.scorer_calls(), written.len() + 1);
    assert_eq!(summary.baseline_score, Some(0.5));
    for candidate in &written {
        assert!(candidate.score.is_some(), "{} scored", candidate.label);
        assert_eq!(
            candidate.score_delta,
            candidate.score.map(|s| s - 0.5),
            "{} delta",
            candidate.label
        );
    }
    // The first candidate scored below the baseline, so it is not a win; the
    // later ones cleared the margin and are ranked best first.
    assert_eq!(written[0].score, Some(0.4));
    assert_eq!(written[0].score_win, Some(false));
    let winners = summary.winners();
    assert!(!winners.is_empty(), "expected at least one scorer win");
    assert!(winners.iter().all(|w| w.score == Some(0.6)));
}

/// Criterion: candidate metadata records the selected neurons / synapses and
/// the strategy — and the written creature moves those genes and no others.
#[test]
fn candidate_metadata_names_exactly_the_genes_that_moved() {
    let fixture = Fixture::new(&["0.5"]);
    let out_name = "out";
    let summary = blocks(
        &fixture,
        out_name,
        &BlockPlan {
            strategies: vec![
                BlockStrategy::Global,
                BlockStrategy::Neuron,
                BlockStrategy::OutputHead,
            ],
            blocks_per_strategy: 1,
            ..BlockPlan::default()
        },
        false,
        false,
    );
    let out = fixture.root.join(out_name);

    // blocks.json is the artefact a downstream tool reads — assert off it.
    let text = fs::read_to_string(out.join("blocks.json")).expect("blocks.json written");
    let reloaded: BlocksSummary = serde_json::from_str(&text).expect("blocks.json parses");
    assert_eq!(reloaded.candidates.len(), summary.candidates.len());
    assert!(text.contains("\"strategy\": \"outputHead\""), "{text}");

    let source = load_forward_only_creature(&fixture.creature).expect("source creature");
    let plank = BackpropConfig::default().plank_constant;
    for candidate in &reloaded.candidates {
        let Some(relative) = &candidate.candidate else {
            continue;
        };
        let applied = load_forward_only_creature(&out.join(relative)).expect("candidate creature");
        for (i, neuron) in applied.neurons.iter().enumerate() {
            if (neuron.bias - source.neurons[i].bias).abs() >= plank {
                assert!(
                    candidate.neurons.contains(&neuron.uuid),
                    "{} moved {} without naming it",
                    candidate.label,
                    neuron.uuid
                );
            }
        }
        for (i, synapse) in applied.synapses.iter().enumerate() {
            if (synapse.weight - source.synapses[i].weight).abs() >= plank {
                assert!(
                    candidate
                        .synapses
                        .iter()
                        .any(|s| s.from_uuid == synapse.from_uuid && s.to_uuid == synapse.to_uuid),
                    "{} moved {}→{} without naming it",
                    candidate.label,
                    synapse.from_uuid,
                    synapse.to_uuid
                );
            }
        }
        // MSE ran, so every written candidate carries its own measurement.
        assert!(candidate.train_mse.is_some(), "{} mse", candidate.label);
        assert!(
            candidate.mse_delta.is_some(),
            "{} mse delta",
            candidate.label
        );
    }

    // Every record's gene count matches the genes it names, and each synapse
    // carries its export index so parallel edges stay distinguishable.
    for candidate in &reloaded.candidates {
        assert_eq!(
            candidate.gene_count,
            candidate.neurons.len() + candidate.synapses.len(),
            "{} gene count",
            candidate.label
        );
        for synapse in &candidate.synapses {
            let export = &source.synapses[synapse.index];
            assert_eq!(export.from_uuid, synapse.from_uuid);
            assert_eq!(export.to_uuid, synapse.to_uuid);
        }
    }

    // The output-head block is confined to the output side of the graph.
    let head = reloaded
        .candidates
        .iter()
        .find(|c| c.strategy == BlockStrategy::OutputHead)
        .expect("output head block");
    assert_eq!(head.neurons, vec!["o1".to_string()]);
    assert!(head.synapses.iter().all(|s| s.to_uuid == "o1"));
    assert_eq!(head.hidden_biases, 0);
}

/// Every block candidate records the size of the update it actually wrote, so
/// a blockwise run can correlate the realised move with `scoreDelta` (#109).
#[test]
fn every_block_records_its_realised_update_norm() {
    let fixture = Fixture::new(&["0.5"]);
    let out_name = "out-norms";
    let summary = blocks(
        &fixture,
        out_name,
        &BlockPlan {
            strategies: vec![BlockStrategy::Global, BlockStrategy::Neuron],
            blocks_per_strategy: 2,
            ..BlockPlan::default()
        },
        false,
        true,
    );
    let text = fs::read_to_string(fixture.root.join(out_name).join("blocks.json"))
        .expect("blocks.json written");
    let reloaded: BlocksSummary = serde_json::from_str(&text).expect("blocks.json parses");
    assert_eq!(reloaded.candidates.len(), summary.candidates.len());

    let mut measured = 0usize;
    for candidate in &reloaded.candidates {
        let moved = candidate.hidden_biases
            + candidate.output_biases
            + candidate.hidden_weights
            + candidate.output_weights;
        assert_eq!(
            candidate.update.total.changed, moved,
            "{}: the norms and the movement counts must agree",
            candidate.label
        );
        assert_eq!(
            candidate.update.biases.changed + candidate.update.weights.changed,
            moved,
            "{}: bias / weight must partition the update",
            candidate.label
        );
        if moved > 0 {
            assert!(
                candidate.update.total.l2 > 0.0,
                "{}: a block that moved genes has a non-zero update",
                candidate.label
            );
            assert!(candidate.update.total.l1 >= candidate.update.total.l2);
            measured += 1;
        }
    }
    assert!(measured > 0, "at least one block moved a gene");
}
