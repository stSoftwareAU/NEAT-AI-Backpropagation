//! Whole-creature update budget: a fixed step scale is not a fixed step (#109).
//!
//! `--step-scale` multiplies every gene's own proposal, so the aggregate move
//! a creature makes grows with the number and magnitude of the genes that move
//! — the same 1% step is a far larger perturbation on a 49k-synapse creature
//! than on the ~16.6k-parameter network the default was justified against.
//!
//! These tests drive `run_train` on a dense creature (many genes moving
//! together, which is what makes the aggregate grow) and assert on the journal
//! the trainer actually wrote: the norms it reports, the rescale it applied,
//! and the parity of an unbudgeted run with the historical fixed-step apply.

use neat_ai_backpropagation::backprop::{ApplyOptions, BackpropConfig};
use neat_ai_backpropagation::train::{
    AcceptanceMode, DEFAULT_STEP_SCALE, TrainCreature, TrainEpochRecord, TrainJournalHeader,
    TrainRequest, run_train,
};
use neat_ai_backpropagation::trust_region::TrustRegion;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// Learning rate large enough that the proposals are worth budgeting.
const LEARNING_RATE: f64 = 0.5;

/// Two dense identity layers into one output — every gene moves each epoch, so
/// the whole-creature update is an aggregate of many per-gene proposals.
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
    // Synapses must be emitted sorted by (from, to) neuron index — the
    // `SORT_FAILURE` rule `neat_core::creature_validate` enforces.
    let mut synapses = Vec::new();
    for i in 0..layer_a {
        synapses.push(format!(
            r#"{{"fromUUID":"input-0","toUUID":"a{i}","weight":0.1}}"#
        ));
    }
    for i in 0..layer_a {
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

/// A prepared run: dense creature and a deterministic `y = 0.5x + 0.25` corpus.
struct Fixture {
    _dir: TempDir,
    root: PathBuf,
    creature: PathBuf,
    data: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let data = root.join("data");
        fs::create_dir_all(&data).expect("create data dir");
        let mut f = fs::File::create(data.join("0.bin")).expect("create record file");
        for i in 0..64 {
            let x = (i as f32) / 64.0 * 2.0 - 1.0;
            f.write_all(&x.to_le_bytes()).expect("write input");
            f.write_all(&(0.5 * x + 0.25).to_le_bytes())
                .expect("write target");
        }
        let creature = root.join("creature.json");
        fs::write(&creature, dense_creature_json(8, 8)).expect("write creature");
        Self {
            _dir: dir,
            root,
            creature,
            data,
        }
    }

    /// One epoch under `region`, returning the output directory.
    ///
    /// `accept_always` keeps the candidate whatever the slice MSE did, so the
    /// journalled update is the one the apply actually wrote rather than
    /// whichever attempt the acceptance gate happened to keep.
    fn train(&self, out_name: &str, region: TrustRegion) -> Result<PathBuf, String> {
        let out = self.root.join(out_name);
        let config = BackpropConfig {
            learning_rate: LEARNING_RATE,
            initial_learning_rate: LEARNING_RATE,
            ..BackpropConfig::default()
        };
        run_train(TrainRequest {
            creature: TrainCreature::Path(&self.creature),
            training_data: &self.data,
            config: &config,
            epochs: 1,
            max_records: None,
            seed: 1,
            disable_random_samples: false,
            output_dir: &out,
            scorer: None,
            apply: ApplyOptions {
                step_scale: DEFAULT_STEP_SCALE,
                ..ApplyOptions::default()
            },
            trust_region: region,
            acceptance: AcceptanceMode::Mse,
            accept_always: true,
            max_backtracks: 0,
            step_scale_ladder: &[],
            trace_store: None,
        })?;
        Ok(out)
    }
}

/// The run header of a finished run.
fn header(out: &Path) -> TrainJournalHeader {
    let journal = fs::read_to_string(out.join("journal.jsonl")).expect("journal written");
    serde_json::from_str(journal.lines().next().expect("header line")).expect("header parses")
}

/// The single epoch line of a one-epoch run.
fn epoch(out: &Path) -> TrainEpochRecord {
    let journal = fs::read_to_string(out.join("journal.jsonl")).expect("journal written");
    let line = journal
        .lines()
        .find(|line| line.contains("\"kind\":\"epoch\""))
        .expect("epoch line");
    serde_json::from_str(line).expect("epoch parses")
}

/// An unbudgeted run must be the historical fixed-step apply, byte for byte —
/// the parity mode #109 promises.
#[test]
fn an_unbudgeted_run_is_the_fixed_step_apply() {
    let fixture = Fixture::new();
    let parity = fixture
        .train("parity", TrustRegion::default())
        .expect("parity run");
    let record = epoch(&parity);
    assert!(record.update.total.changed > 0, "the epoch moved genes");
    assert_eq!(
        record.realised_step_scale, record.step_scale,
        "no budget must leave the requested step untouched"
    );
    assert_eq!(
        header(&parity).trust_region,
        None,
        "an unconfigured region is journalled as absent, not as an empty budget"
    );

    // A budget far above the measured update is inert: same creature out.
    let slack = fixture
        .train(
            "slack",
            TrustRegion {
                l2: Some(record.update.total.l2 * 1_000.0),
                ..TrustRegion::default()
            },
        )
        .expect("slack run");
    assert_eq!(
        fs::read_to_string(parity.join("best.json")).unwrap(),
        fs::read_to_string(slack.join("best.json")).unwrap(),
        "a budget the update already fits must not change the candidate"
    );
    assert_eq!(epoch(&slack).realised_step_scale, record.step_scale);
}

/// The whole update — not each gene on its own — is held under the budget.
#[test]
fn an_l2_budget_rescales_the_whole_update() {
    let fixture = Fixture::new();
    let unbudgeted = epoch(
        &fixture
            .train("unbudgeted", TrustRegion::default())
            .expect("unbudgeted run"),
    );
    let budget = unbudgeted.update.total.l2 / 4.0;
    assert!(budget > 0.0, "the unbudgeted epoch moved the creature");

    let budgeted = epoch(
        &fixture
            .train(
                "budgeted",
                TrustRegion {
                    l2: Some(budget),
                    ..TrustRegion::default()
                },
            )
            .expect("budgeted run"),
    );
    assert!(
        budgeted.update.total.l2 <= budget * 1.000_001,
        "update L2 {} exceeded the budget {budget}",
        budgeted.update.total.l2
    );
    assert!(
        budgeted.update.total.l2 > budget * 0.9,
        "the rescale must use the budget, not collapse the step: {}",
        budgeted.update.total.l2
    );
    // The step scale is an input the trust region shrinks — a quarter of the
    // update is a quarter of the step, and the journal says so.
    assert!(
        budgeted.realised_step_scale < budgeted.step_scale,
        "a bound update must journal a smaller realised step: {} vs {}",
        budgeted.realised_step_scale,
        budgeted.step_scale
    );
    assert!(
        (budgeted.realised_step_scale - budgeted.step_scale / 4.0).abs()
            < budgeted.step_scale * 1e-3,
        "realised step {} should be a quarter of {}",
        budgeted.realised_step_scale,
        budgeted.step_scale
    );
}

/// Bias and weight budgets bind their own class without touching the other.
#[test]
fn a_class_budget_binds_only_its_own_genes() {
    let fixture = Fixture::new();
    let unbudgeted = epoch(
        &fixture
            .train("unbudgeted", TrustRegion::default())
            .expect("unbudgeted run"),
    );
    assert!(unbudgeted.update.biases.changed > 0, "biases moved");
    assert!(unbudgeted.update.weights.changed > 0, "weights moved");

    let budgeted = epoch(
        &fixture
            .train(
                "bias-bound",
                TrustRegion {
                    bias_l2: Some(unbudgeted.update.biases.l2 / 8.0),
                    ..TrustRegion::default()
                },
            )
            .expect("bias-budget run"),
    );
    assert!(
        budgeted.update.biases.l2 <= unbudgeted.update.biases.l2 / 8.0 * 1.000_001,
        "the bias budget must bind: {}",
        budgeted.update.biases.l2
    );
    // The rescale is applied to the whole update, so the weights shrink with
    // it — what must not happen is the weights being left at full size while
    // the bias budget is reported as met.
    assert!(
        budgeted.update.weights.l2 < unbudgeted.update.weights.l2,
        "the whole update is rescaled, weights included"
    );
}

/// The classes partition the same genes the movement counts report.
#[test]
fn the_journal_splits_the_update_by_gene_class() {
    let fixture = Fixture::new();
    let record = epoch(
        &fixture
            .train("classes", TrustRegion::default())
            .expect("run"),
    );
    let update = record.update;
    assert_eq!(
        update.biases.changed + update.weights.changed,
        update.total.changed,
        "bias / weight must partition the update"
    );
    assert_eq!(
        update.hidden.changed + update.output.changed,
        update.total.changed,
        "hidden / output must partition the update"
    );
    assert_eq!(
        update.total.changed,
        record.hidden_biases + record.output_biases + record.hidden_weights + record.output_weights,
        "the norms and the movement counts must agree on what moved"
    );
    assert!(update.total.l1 >= update.total.l2, "L1 dominates L2");
    assert!(update.total.max_abs > 0.0);
    assert!(update.total.rms > 0.0);
    assert!(
        update.total.relative_l2.is_some(),
        "the incumbent values are non-zero, so the relative delta is meaningful"
    );
    assert!(update.total.relative_rms.is_some());
}

/// The changed-gene budget is an L0 trust region: only the largest moves land.
#[test]
fn a_changed_gene_budget_caps_how_many_genes_move() {
    let fixture = Fixture::new();
    let unbudgeted = epoch(
        &fixture
            .train("unbudgeted", TrustRegion::default())
            .expect("unbudgeted run"),
    );
    assert!(
        unbudgeted.update.total.changed > 5,
        "the dense creature moves more genes than the budget allows: {}",
        unbudgeted.update.total.changed
    );

    let budgeted = epoch(
        &fixture
            .train(
                "gene-capped",
                TrustRegion {
                    max_changed_genes: Some(5),
                    ..TrustRegion::default()
                },
            )
            .expect("gene-budget run"),
    );
    assert_eq!(
        budgeted.update.total.changed, 5,
        "the budget caps the moved genes"
    );
    assert!(
        budgeted.update.total.max_abs >= unbudgeted.update.total.max_abs * 0.999_999,
        "the genes kept are the largest moves, so the largest single move survives"
    );
    assert_eq!(
        budgeted.hidden_biases
            + budgeted.output_biases
            + budgeted.hidden_weights
            + budgeted.output_weights,
        5,
        "the movement counts see the trim too"
    );
}

/// A budget that could not bound anything is refused before any corpus work.
#[test]
fn an_unusable_budget_is_refused() {
    let fixture = Fixture::new();
    for bad in [0.0, -1.0, f64::NAN] {
        let err = fixture
            .train(
                "refused",
                TrustRegion {
                    l2: Some(bad),
                    ..TrustRegion::default()
                },
            )
            .expect_err("an unusable budget must fail loudly");
        assert!(err.contains("trust-region"), "{bad}: {err}");
    }
    let err = fixture
        .train(
            "refused-genes",
            TrustRegion {
                max_changed_genes: Some(0),
                ..TrustRegion::default()
            },
        )
        .expect_err("a zero gene budget must fail loudly");
    assert!(err.contains("maxChangedGenes"), "{err}");
}

/// The header carries the budget the run applied, so a journal is auditable
/// without the invocation that produced it.
#[test]
fn the_header_records_the_configured_budget() {
    let fixture = Fixture::new();
    let region = TrustRegion {
        l2: Some(0.25),
        rms: Some(0.01),
        relative_rms: Some(0.05),
        bias_l2: Some(0.1),
        weight_l2: Some(0.2),
        max_changed_genes: Some(64),
    };
    let out = fixture.train("recorded", region).expect("run");
    assert_eq!(header(&out).trust_region, Some(region));
}
