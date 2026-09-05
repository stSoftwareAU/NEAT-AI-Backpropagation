//! Issue #107: gradient diagnostics stratified by gene class and squash.
//!
//! One whole-creature sign-agreement percentage cannot say *where* a backprop
//! proposal is trustworthy. These tests drive `gradient-check` over a
//! heterogeneous creature — mixed squashes, an aggregate neuron, a deep path,
//! a dead branch — and assert the artefact carries the facets, the applied
//! ground truth, a deterministic seed contract and a ranked best / worst list.

use neat_ai_backpropagation::backprop::BackpropConfig;
use neat_ai_backpropagation::gene_facets::facet;
use neat_ai_backpropagation::gradient_check::{
    GRADIENT_CHECK_SCHEMA, GeneClass, GradientCheckRequest, GradientCheckSummary,
    run_gradient_check, summary_text,
};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// Two inputs, four hidden neurons and an aggregate output.
///
/// `h1` (TANH) is fed by both inputs; `h2` (IDENTITY) sits behind it, so the
/// output has a two-hop and a three-hop path into it. `dead` is a ReLU held
/// below zero by a large negative bias — the low-activity / saturated case.
const MIXED: &str = r#"{
  "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
  "neurons":[
    {"type":"hidden","uuid":"h1","bias":0.1,"squash":"TANH"},
    {"type":"hidden","uuid":"h2","bias":-0.2,"squash":"IDENTITY"},
    {"type":"hidden","uuid":"dead","bias":-50.0,"squash":"ReLU"},
    {"type":"hidden","uuid":"wide","bias":0.05,"squash":"LOGISTIC"},
    {"type":"output","uuid":"o1","bias":0.25,"squash":"MAXIMUM"}
  ],
  "synapses":[
    {"fromUUID":"input-0","toUUID":"h1","weight":0.75},
    {"fromUUID":"input-1","toUUID":"h1","weight":-0.5},
    {"fromUUID":"input-0","toUUID":"dead","weight":0.25},
    {"fromUUID":"input-1","toUUID":"wide","weight":1.5},
    {"fromUUID":"input-0","toUUID":"wide","weight":0.5},
    {"fromUUID":"h1","toUUID":"h2","weight":1.25},
    {"fromUUID":"h1","toUUID":"o1","weight":0.9},
    {"fromUUID":"h2","toUUID":"o1","weight":-0.75},
    {"fromUUID":"dead","toUUID":"o1","weight":0.4},
    {"fromUUID":"wide","toUUID":"o1","weight":0.6}
  ]
}"#;

/// Write the creature plus an eight-record two-input / one-output corpus.
fn fixture() -> (TempDir, PathBuf, PathBuf) {
    let dir = tempdir().unwrap();
    let creature_path = dir.path().join("creature.json");
    fs::write(&creature_path, MIXED).unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let mut file = fs::File::create(data.join("0.bin")).unwrap();
    for step in 0..8u32 {
        let a = step as f32 * 0.25 - 1.0;
        let b = 1.0 - step as f32 * 0.125;
        let target = 0.5 * a - 0.25 * b + 0.1;
        for value in [a, b, target] {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }
    (dir, creature_path, data)
}

/// Run `gradient-check` over the fixture with the given seed and caps.
fn probe(
    creature: &Path,
    data: &Path,
    out: &Path,
    seed: u64,
    biases: usize,
    weights: usize,
) -> GradientCheckSummary {
    let config = BackpropConfig::default();
    run_gradient_check(GradientCheckRequest {
        creature,
        training_data: data,
        config: &config,
        max_records: Some(8),
        seed,
        sample_biases: biases,
        sample_weights: weights,
        fd_eps: 1e-4,
        step_scale: 1.0,
        outputs_only: false,
        hidden_only: false,
        facet_min_scored: 2,
        rank_limit: 3,
        output_dir: out,
    })
    .expect("gradient-check must run over the mixed creature")
}

#[test]
fn every_gene_carries_its_class_and_squash_facets() {
    let (dir, creature, data) = fixture();
    let out = dir.path().join("out");
    let summary = probe(&creature, &data, &out, 1, 16, 16);

    assert_eq!(summary.schema_version, GRADIENT_CHECK_SCHEMA);
    assert_eq!(summary.creature.neurons, 5);
    assert_eq!(summary.creature.synapses, 10);
    assert!(!summary.genes.is_empty(), "the probe must sample genes");
    assert_eq!(summary.sampled, summary.genes.len());

    for gene in &summary.genes {
        assert!(
            !gene.attributes.squash.is_empty(),
            "gene {} carries no squash",
            gene.id
        );
        assert!(
            gene.attributes.depth >= 1,
            "gene {} is not reachable from an input",
            gene.id
        );
        assert!(
            !gene.attributes.activity.is_empty(),
            "gene {} carries no activation bucket",
            gene.id
        );
    }

    // The aggregate output must be labelled as such wherever it was sampled.
    let aggregate_rows: Vec<_> = summary
        .genes
        .iter()
        .filter(|g| g.attributes.aggregate)
        .collect();
    assert!(
        aggregate_rows
            .iter()
            .all(|g| g.attributes.squash == "MAXIMUM"),
        "only the MAXIMUM output is an aggregate"
    );

    // Every facet the issue asks for is reported, and every facet accounts for
    // the whole sample.
    for name in [
        facet::GENE_KIND,
        facet::ROLE,
        facet::CLASS,
        facet::SQUASH,
        facet::AGGREGATE,
        facet::DEPTH,
        facet::FAN_IN,
        facet::FAN_OUT,
        facet::ACTIVITY,
        facet::PROPOSAL_MAGNITUDE,
    ] {
        let buckets: Vec<_> = summary
            .by_facet
            .iter()
            .filter(|s| s.facet == name)
            .collect();
        assert!(!buckets.is_empty(), "facet {name} is missing");
        let sampled: usize = buckets.iter().map(|s| s.sampled).sum();
        assert_eq!(
            sampled,
            summary.genes.len(),
            "facet {name} lost genes: {sampled} of {}",
            summary.genes.len()
        );
    }

    // More than one squash was actually sampled — the stratification has
    // something to contrast.
    let squashes: std::collections::BTreeSet<&str> = summary
        .by_facet
        .iter()
        .filter(|s| s.facet == facet::SQUASH)
        .map(|s| s.bucket.as_str())
        .collect();
    assert!(
        squashes.len() > 1,
        "expected several squashes, got {squashes:?}"
    );
}

#[test]
fn gradient_error_compares_the_proposal_against_the_finite_difference() {
    let (dir, creature, data) = fixture();
    let out = dir.path().join("out");
    let summary = probe(&creature, &data, &out, 7, 16, 16);
    let scale = summary.learning_rate * summary.step_scale;
    assert!(scale > 0.0, "the fixture must run with a real step");

    for gene in &summary.genes {
        let implied = -gene.proposal_delta / scale;
        assert!(
            (gene.proposal_grad - implied).abs() < 1e-9,
            "proposal gradient must invert Δ through lr · step for {}",
            gene.id
        );
        assert!(
            (gene.grad_abs_error - (gene.proposal_grad - gene.fd_grad).abs()).abs() < 1e-9,
            "absolute gradient error must be |proposal − fd| for {}",
            gene.id
        );
        let expected_rel = gene.grad_abs_error / gene.proposal_grad.abs().max(gene.fd_grad.abs());
        assert!(
            (gene.grad_rel_error.expect("a moved gene has a gradient") - expected_rel).abs() < 1e-9,
            "relative gradient error must scale by the larger gradient for {}",
            gene.id
        );
    }
}

#[test]
fn applied_proposals_are_measured_not_predicted() {
    let (dir, creature, data) = fixture();
    let out = dir.path().join("out");
    let summary = probe(&creature, &data, &out, 7, 16, 16);

    for gene in &summary.genes {
        let predicted = gene.fd_grad * gene.proposal_delta;
        assert!(
            (gene.predicted_delta_mse - predicted).abs() < 1e-12,
            "predicted ΔMSE must be fd_grad · Δ for {}",
            gene.id
        );
        assert_eq!(
            gene.improved,
            gene.actual_delta_mse < 0.0,
            "improved must follow the measured MSE change for {}",
            gene.id
        );
    }
    assert_eq!(
        summary.improved,
        summary.genes.iter().filter(|g| g.improved).count()
    );
    assert!(summary.improved_pct >= 0.0 && summary.improved_pct <= 100.0);
}

#[test]
fn the_same_seed_reproduces_the_artefact_byte_for_byte() {
    let (dir, creature, data) = fixture();
    let first = dir.path().join("first");
    let second = dir.path().join("second");
    let a = probe(&creature, &data, &first, 42, 8, 8);
    let b = probe(&creature, &data, &second, 42, 8, 8);

    assert_eq!(
        fs::read_to_string(first.join("gradient-check.json")).unwrap(),
        fs::read_to_string(second.join("gradient-check.json")).unwrap(),
        "the same seed must reproduce the summary"
    );
    assert_eq!(
        fs::read_to_string(first.join("genes.jsonl")).unwrap(),
        fs::read_to_string(second.join("genes.jsonl")).unwrap(),
        "the same seed must reproduce the gene rows"
    );
    assert_eq!(a.seed, 42);
    assert_eq!(a.sampled, b.sampled);
    assert_eq!(summary_text(&a), summary_text(&b));
}

#[test]
fn the_sample_caps_bound_the_run() {
    let (dir, creature, data) = fixture();
    let out = dir.path().join("out");
    let summary = probe(&creature, &data, &out, 3, 1, 2);
    let biases = summary
        .genes
        .iter()
        .filter(|g| matches!(g.class, GeneClass::HiddenBias | GeneClass::OutputBias))
        .count();
    let weights = summary.genes.len() - biases;
    assert!(biases <= 1, "the bias cap must hold: {biases}");
    assert!(weights <= 2, "the weight cap must hold: {weights}");
    assert_eq!(
        summary.sampled,
        summary.genes.len(),
        "the summary must count what it probed"
    );
}

#[test]
fn best_and_worst_classes_are_ranked_with_evidence() {
    let (dir, creature, data) = fixture();
    let out = dir.path().join("out");
    let summary = probe(&creature, &data, &out, 11, 16, 16);

    assert!(
        !summary.best_classes.is_empty(),
        "a mixed creature must rank at least one bucket"
    );
    assert!(!summary.worst_classes.is_empty());
    for ranked in summary.best_classes.iter().chain(&summary.worst_classes) {
        assert!(
            ranked.scored >= 2,
            "{}={} was ranked on {} scored genes",
            ranked.facet,
            ranked.bucket,
            ranked.scored
        );
        assert!(
            summary.by_facet.contains(ranked),
            "ranking must cite a facet"
        );
    }
    assert!(
        summary.best_classes[0].sign_agree_pct >= summary.worst_classes[0].sign_agree_pct,
        "best must not score below worst"
    );
    assert!(
        summary.best_classes.len() <= 3,
        "rank limit must be honoured"
    );
    for ranked in &summary.worst_classes {
        assert!(
            !summary
                .best_classes
                .iter()
                .any(|b| b.facet == ranked.facet && b.bucket == ranked.bucket),
            "{}={} was reported as both best and worst",
            ranked.facet,
            ranked.bucket
        );
    }
}

#[test]
fn a_sample_too_thin_to_rank_says_so_rather_than_inventing_a_winner() {
    let (dir, creature, data) = fixture();
    let out = dir.path().join("out");
    let config = BackpropConfig::default();
    let summary = run_gradient_check(GradientCheckRequest {
        creature: &creature,
        training_data: &data,
        config: &config,
        max_records: Some(8),
        seed: 2,
        sample_biases: 4,
        sample_weights: 4,
        fd_eps: 1e-4,
        step_scale: 1.0,
        outputs_only: false,
        hidden_only: false,
        // No bucket can hold 500 genes, so nothing is rankable.
        facet_min_scored: 500,
        rank_limit: 5,
        output_dir: &out,
    })
    .expect("the probe must still run when nothing is rankable");

    assert!(summary.best_classes.is_empty());
    assert!(summary.worst_classes.is_empty());
    let text = fs::read_to_string(out.join("summary.txt")).unwrap();
    assert!(
        text.contains("no bucket carried enough scored genes to rank"),
        "an unrankable run must say so:\n{text}"
    );
}

/// A synapse pointing at a UUID no neuron owns must fail loudly rather than
/// quietly shrinking the sample.
#[test]
fn a_dangling_synapse_target_fails_loudly() {
    let (dir, _creature, data) = fixture();
    let broken_path = dir.path().join("broken.json");
    fs::write(
        &broken_path,
        MIXED.replace(
            r#"{"fromUUID":"h1","toUUID":"o1","weight":0.9}"#,
            r#"{"fromUUID":"h1","toUUID":"ghost","weight":0.9}"#,
        ),
    )
    .unwrap();
    let config = BackpropConfig::default();
    let err = run_gradient_check(GradientCheckRequest {
        creature: &broken_path,
        training_data: &data,
        config: &config,
        max_records: Some(8),
        seed: 1,
        sample_biases: 4,
        sample_weights: 4,
        fd_eps: 1e-4,
        step_scale: 1.0,
        outputs_only: false,
        hidden_only: false,
        facet_min_scored: 2,
        rank_limit: 3,
        output_dir: &dir.path().join("broken-out"),
    })
    .expect_err("a dangling synapse target must not be probed silently");
    assert!(
        err.contains("ghost"),
        "the error must name the target: {err}"
    );
}

#[test]
fn the_run_leaves_a_concise_summary_for_an_unattended_reader() {
    let (dir, creature, data) = fixture();
    let out = dir.path().join("out");
    let summary = probe(&creature, &data, &out, 5, 16, 16);

    let text = fs::read_to_string(out.join("summary.txt")).unwrap();
    assert_eq!(text, summary_text(&summary));
    assert!(text.contains("gradient-check v"), "{text}");
    assert!(text.contains("signAgree="), "{text}");
    assert!(text.contains("improved="), "{text}");
    assert!(text.contains("best :"), "{text}");
    assert!(text.contains("worst:"), "{text}");
    assert!(
        text.lines().count() <= 12,
        "the summary must stay concise:\n{text}"
    );
}
