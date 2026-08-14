//! Parity dump / diff for Rust ↔ TypeScript backpropagation.

use crate::backprop::{
    BackpropConfig, LearningSignal, apply_learnings, calculate_learning_rate, nearly_equal,
};
use crate::propagate_layout::accumulate_creature_learning_report;
use neat_core::{CreatureExport, compile_creature, parse_creature_json};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// One export neuron's accumulated and proposed bias.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NeuronCompare {
    /// Neuron UUID (export / wire id).
    pub uuid: String,
    /// Accumulation count.
    pub bias_count: f64,
    /// Sum of adjusted bias contributions.
    pub total_adjusted_bias: f64,
    /// Bias before apply.
    pub current_bias: f64,
    /// Bias after [`apply_learnings`].
    pub proposed_bias: f64,
}

/// One export synapse's accumulated and proposed weight.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SynapseCompare {
    /// Source UUID.
    #[serde(rename = "fromUUID")]
    pub from_uuid: String,
    /// Destination UUID.
    #[serde(rename = "toUUID")]
    pub to_uuid: String,
    /// Accumulation count.
    pub count: f64,
    /// Weight before apply.
    pub current_weight: f64,
    /// Weight after [`apply_learnings`].
    pub proposed_weight: f64,
}

/// Side-by-side dump used by the `compare` CLI and the Deno dual-run harness.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CompareDump {
    /// Crate version that wrote the dump (`CARGO_PKG_VERSION` on the Rust side).
    pub version: String,
    /// Records consumed.
    pub records: u64,
    /// Forward-pass MSE over those records.
    pub mse: f64,
    /// Learning rate used for proposals.
    pub learning_rate: f64,
    /// Per-neuron rows (export order).
    pub neurons: Vec<NeuronCompare>,
    /// Per-synapse rows (export order).
    pub synapses: Vec<SynapseCompare>,
}

/// Build a dump from an accumulated signal and the creature it was measured on.
pub fn build_compare_dump(
    creature: &CreatureExport,
    signal: &LearningSignal,
    config: &BackpropConfig,
    learning_rate: f64,
    mse: f64,
    records: u64,
    version: impl Into<String>,
) -> CompareDump {
    let applied = apply_learnings(creature, signal, config, learning_rate);
    let mut neurons = Vec::with_capacity(creature.neurons.len());
    for (i, n) in creature.neurons.iter().enumerate() {
        let sig = signal.biases.get(i).cloned().unwrap_or_default();
        neurons.push(NeuronCompare {
            uuid: n.uuid.clone(),
            bias_count: sig.count,
            total_adjusted_bias: sig.total_adjusted_bias,
            current_bias: n.bias,
            proposed_bias: applied.neurons.get(i).map(|a| a.bias).unwrap_or(n.bias),
        });
    }
    let mut synapses = Vec::with_capacity(creature.synapses.len());
    for (i, s) in creature.synapses.iter().enumerate() {
        let sig = signal.weights.get(i).cloned().unwrap_or_default();
        synapses.push(SynapseCompare {
            from_uuid: s.from_uuid.clone(),
            to_uuid: s.to_uuid.clone(),
            count: sig.count,
            current_weight: s.weight,
            proposed_weight: applied
                .synapses
                .get(i)
                .map(|a| a.weight)
                .unwrap_or(s.weight),
        });
    }
    CompareDump {
        version: version.into(),
        records,
        mse,
        learning_rate,
        neurons,
        synapses,
    }
}

/// Run accumulate on a creature + data dir and write a compare dump.
pub fn run_compare(
    creature_path: &Path,
    training_data: &Path,
    config: &BackpropConfig,
    max_records: Option<u64>,
    seed: u64,
    out_path: &Path,
) -> Result<CompareDump, String> {
    let text = fs::read_to_string(creature_path).map_err(|e| e.to_string())?;
    let creature = parse_creature_json(&text).map_err(|e| e.to_string())?;
    let mut network = compile_creature(&creature).map_err(|e| e.to_string())?;
    let mut rng = StdRng::seed_from_u64(seed);
    let report = accumulate_creature_learning_report(
        &creature,
        &mut network,
        training_data,
        config,
        max_records,
        &mut rng,
    )?;
    let lr = calculate_learning_rate(config, 0, None);
    let dump = build_compare_dump(
        &creature,
        &report.learning,
        config,
        lr,
        report.mse,
        report.records,
        env!("CARGO_PKG_VERSION"),
    );
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(&dump).map_err(|e| e.to_string())?;
    fs::write(out_path, json).map_err(|e| e.to_string())?;
    let _ = network;
    Ok(dump)
}

/// Human-readable mismatch between two dumps.
#[derive(Debug, Clone)]
pub struct CompareMismatch {
    /// Where the values disagreed.
    pub path: String,
    /// Left-hand value.
    pub left: f64,
    /// Right-hand value.
    pub right: f64,
}

/// Summary of a production parity diff.
#[derive(Debug, Clone)]
pub struct CompareDiffReport {
    /// Field mismatches on the shared (both-sides-updated) surface.
    pub mismatches: Vec<CompareMismatch>,
    /// Neurons Rust updated that TypeScript/WASM left at count 0.
    pub rust_only_neurons: usize,
    /// Neurons TypeScript updated that Rust left at count 0.
    pub other_only_neurons: usize,
    /// Neurons both sides accumulated.
    pub overlap_neurons: usize,
    /// Synapses Rust updated that TypeScript left at count 0.
    pub rust_only_synapses: usize,
    /// Synapses TypeScript updated that Rust left at count 0.
    pub other_only_synapses: usize,
    /// Synapses both sides accumulated.
    pub overlap_synapses: usize,
}

/// Diff two dumps.
///
/// Record counts and MSE always compare. By default only neurons/synapses
/// that **both** sides accumulated (`count > 0`) are compared — production
/// creatures contain IF/MIN/MAX aggregates where neat-core WASM stops and
/// TypeScript custom-propagate does not continue the reverse-topo loop,
/// while this crate linearises those aggregates (Lamarck issue #83).
///
/// `strict` also requires matching counts on every gene (including
/// Rust-only aggregate continuation).
pub fn diff_compare_dumps(
    rust: &CompareDump,
    other: &CompareDump,
    strict: bool,
) -> Result<CompareDiffReport, String> {
    if rust.records != other.records {
        return Err(format!(
            "record count mismatch: rust={} other={}",
            rust.records, other.records
        ));
    }
    let mut mismatches = Vec::new();
    push_if_diff(&mut mismatches, "mse", rust.mse, other.mse);

    let other_neurons: BTreeMap<&str, &NeuronCompare> =
        other.neurons.iter().map(|n| (n.uuid.as_str(), n)).collect();
    let mut rust_only_neurons = 0usize;
    let mut other_only_neurons = 0usize;
    let mut overlap_neurons = 0usize;
    for n in &rust.neurons {
        let Some(rhs) = other_neurons.get(n.uuid.as_str()) else {
            return Err(format!("other dump missing neuron {}", n.uuid));
        };
        let rust_hit = n.bias_count > 0.0;
        let other_hit = rhs.bias_count > 0.0;
        if rust_hit && other_hit {
            overlap_neurons += 1;
            push_if_diff(
                &mut mismatches,
                &format!("neuron[{}].biasCount", n.uuid),
                n.bias_count,
                rhs.bias_count,
            );
            push_if_diff(
                &mut mismatches,
                &format!("neuron[{}].totalAdjustedBias", n.uuid),
                n.total_adjusted_bias,
                rhs.total_adjusted_bias,
            );
            push_if_diff(
                &mut mismatches,
                &format!("neuron[{}].proposedBias", n.uuid),
                n.proposed_bias,
                rhs.proposed_bias,
            );
        } else if rust_hit {
            rust_only_neurons += 1;
            if strict {
                push_if_diff(
                    &mut mismatches,
                    &format!("neuron[{}].biasCount", n.uuid),
                    n.bias_count,
                    rhs.bias_count,
                );
            }
        } else if other_hit {
            other_only_neurons += 1;
            if strict {
                push_if_diff(
                    &mut mismatches,
                    &format!("neuron[{}].biasCount", n.uuid),
                    n.bias_count,
                    rhs.bias_count,
                );
            }
        }
    }

    let other_syn: BTreeMap<(String, String), &SynapseCompare> = other
        .synapses
        .iter()
        .map(|s| ((s.from_uuid.clone(), s.to_uuid.clone()), s))
        .collect();
    let mut rust_only_synapses = 0usize;
    let mut other_only_synapses = 0usize;
    let mut overlap_synapses = 0usize;
    for s in &rust.synapses {
        let key = (s.from_uuid.clone(), s.to_uuid.clone());
        let Some(rhs) = other_syn.get(&key) else {
            return Err(format!(
                "other dump missing synapse {} -> {}",
                s.from_uuid, s.to_uuid
            ));
        };
        let rust_hit = s.count > 0.0;
        let other_hit = rhs.count > 0.0;
        if rust_hit && other_hit {
            overlap_synapses += 1;
            // Overlap *neuron* bias is the shared-kernel gate. Weight
            // proposals on the same genes diverge once Rust linearises
            // IF/MIN/MAX and extra error flows into those synapses.
            // `--strict` still compares counts and proposed weights.
            if strict {
                push_if_diff(
                    &mut mismatches,
                    &format!("synapse[{}->{}].count", s.from_uuid, s.to_uuid),
                    s.count,
                    rhs.count,
                );
                push_if_diff(
                    &mut mismatches,
                    &format!("synapse[{}->{}].proposedWeight", s.from_uuid, s.to_uuid),
                    s.proposed_weight,
                    rhs.proposed_weight,
                );
            }
        } else if rust_hit {
            rust_only_synapses += 1;
            if strict {
                push_if_diff(
                    &mut mismatches,
                    &format!("synapse[{}->{}].count", s.from_uuid, s.to_uuid),
                    s.count,
                    rhs.count,
                );
            }
        } else if other_hit {
            other_only_synapses += 1;
            if strict {
                push_if_diff(
                    &mut mismatches,
                    &format!("synapse[{}->{}].count", s.from_uuid, s.to_uuid),
                    s.count,
                    rhs.count,
                );
            }
        }
    }
    Ok(CompareDiffReport {
        mismatches,
        rust_only_neurons,
        other_only_neurons,
        overlap_neurons,
        rust_only_synapses,
        other_only_synapses,
        overlap_synapses,
    })
}

fn push_if_diff(out: &mut Vec<CompareMismatch>, path: &str, left: f64, right: f64) {
    if !nearly_equal(left, right) {
        out.push(CompareMismatch {
            path: path.to_string(),
            left,
            right,
        });
    }
}

/// Load a dump written by this CLI or the Deno harness.
pub fn load_compare_dump(path: &Path) -> Result<CompareDump, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backprop::BiasSignal;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::tempdir;

    /// One input feeding two identity outputs (`o1` ×1, `o2` ×2).
    const TWO_OUTPUT: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":2,
      "neurons":[
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"},
        {"type":"output","uuid":"o2","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"o1","weight":1.0},
        {"fromUUID":"input-0","toUUID":"o2","weight":2.0}
      ]
    }"#;

    /// Identity chain `input-0 → h1 → o1`, so every gene accumulates on every
    /// record and the dump carries a non-empty neuron *and* synapse surface.
    const IDENTITY_CHAIN: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
      "neurons":[
        {"type":"hidden","uuid":"h1","bias":0.25,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":-0.5,"squash":"IDENTITY"}
      ],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"h1","weight":0.75},
        {"fromUUID":"h1","toUUID":"o1","weight":1.5}
      ]
    }"#;

    /// Write `creature` plus a one-file `.bin` corpus of `(input, target)`
    /// pairs into a fresh temp directory. Returns `(dir, creature_path,
    /// data_dir)`.
    fn fixture(creature: &str, records: &[(f32, f32)]) -> (tempfile::TempDir, PathBuf, PathBuf) {
        let dir = tempdir().unwrap();
        let creature_path = dir.path().join("creature.json");
        fs::write(&creature_path, creature).unwrap();
        let data_dir = dir.path().join("data");
        fs::create_dir_all(&data_dir).unwrap();
        let mut f = fs::File::create(data_dir.join("0.bin")).unwrap();
        for (input, target) in records {
            f.write_all(&input.to_le_bytes()).unwrap();
            f.write_all(&target.to_le_bytes()).unwrap();
        }
        drop(f);
        (dir, creature_path, data_dir)
    }

    /// Issue #23: the parity dump's file round-trip — `run_compare` writes the
    /// JSON, `load_compare_dump` reads it back — had no coverage, so a serde
    /// rename drift on either side could break Rust ↔ TypeScript parity with a
    /// green suite. Reloading must reproduce the returned dump exactly.
    #[test]
    fn compare_dump_survives_the_file_round_trip() {
        let (dir, creature_path, data_dir) =
            fixture(IDENTITY_CHAIN, &[(1.0, 2.0), (2.0, 0.0), (-1.0, 0.5)]);
        // A nested `--out` also exercises the parent-directory creation.
        let out_path = dir.path().join("dumps").join("rust-compare.json");

        let dump = run_compare(
            &creature_path,
            &data_dir,
            &BackpropConfig::default(),
            None,
            23,
            &out_path,
        )
        .unwrap();
        assert!(out_path.is_file(), "run_compare did not write {out_path:?}");

        let reloaded = load_compare_dump(&out_path).unwrap();

        assert_eq!(
            reloaded, dump,
            "the reloaded dump must equal the one run_compare returned"
        );
        assert_eq!(reloaded.records, 3);
        assert_eq!(
            reloaded.mse, dump.mse,
            "mse must round-trip bit-identically"
        );
        assert_eq!(
            reloaded
                .neurons
                .iter()
                .map(|n| n.uuid.as_str())
                .collect::<Vec<_>>(),
            ["h1", "o1"],
            "neuron UUID keys drifted across the round-trip"
        );
        assert_eq!(
            reloaded
                .synapses
                .iter()
                .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
                .collect::<Vec<_>>(),
            [("input-0", "h1"), ("h1", "o1")],
            "synapse UUID keys drifted across the round-trip"
        );
        assert!(
            reloaded.neurons.iter().any(|n| n.bias_count > 0.0),
            "fixture accumulated nothing — the round-trip would be vacuous"
        );
        assert!(
            diff_compare_dumps(&dump, &reloaded, true)
                .unwrap()
                .mismatches
                .is_empty(),
            "a strict diff of the dump against its own reload must be clean"
        );
    }

    /// The on-disk field names are the wire contract with the Deno harness
    /// (`camelCase`, plus the explicit `fromUUID` / `toUUID` renames). Assert
    /// them on the serialised bytes so a rename fails here, not in production
    /// parity.
    #[test]
    fn dump_file_keeps_the_typescript_wire_field_names() {
        let (dir, creature_path, data_dir) = fixture(IDENTITY_CHAIN, &[(1.0, 2.0), (2.0, 0.0)]);
        let out_path = dir.path().join("rust-compare.json");
        run_compare(
            &creature_path,
            &data_dir,
            &BackpropConfig::default(),
            None,
            23,
            &out_path,
        )
        .unwrap();

        let value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&out_path).unwrap()).unwrap();
        for key in [
            "version",
            "records",
            "mse",
            "learningRate",
            "neurons",
            "synapses",
        ] {
            assert!(
                value.get(key).is_some(),
                "dump is missing top-level `{key}`"
            );
        }
        let neuron = &value["neurons"][0];
        for key in [
            "uuid",
            "biasCount",
            "totalAdjustedBias",
            "currentBias",
            "proposedBias",
        ] {
            assert!(neuron.get(key).is_some(), "neuron row is missing `{key}`");
        }
        let synapse = &value["synapses"][0];
        for key in [
            "fromUUID",
            "toUUID",
            "count",
            "currentWeight",
            "proposedWeight",
        ] {
            assert!(synapse.get(key).is_some(), "synapse row is missing `{key}`");
        }
    }

    /// The diff had only ever been run against an identical dump, so its
    /// mismatch-detection side was never exercised. Perturbing one reloaded
    /// `proposedBias` beyond tolerance must be reported.
    #[test]
    fn diff_reports_a_perturbed_proposed_bias() {
        let (dir, creature_path, data_dir) = fixture(IDENTITY_CHAIN, &[(1.0, 2.0), (2.0, 0.0)]);
        let out_path = dir.path().join("rust-compare.json");
        let dump = run_compare(
            &creature_path,
            &data_dir,
            &BackpropConfig::default(),
            None,
            23,
            &out_path,
        )
        .unwrap();

        let mut other = load_compare_dump(&out_path).unwrap();
        let target = other
            .neurons
            .iter_mut()
            .find(|n| n.bias_count > 0.0)
            .expect("fixture must accumulate at least one neuron bias");
        let uuid = target.uuid.clone();
        let original = target.proposed_bias;
        target.proposed_bias = original + 1.0;

        let report = diff_compare_dumps(&dump, &other, false).unwrap();

        let hit = report
            .mismatches
            .iter()
            .find(|m| m.path == format!("neuron[{uuid}].proposedBias"))
            .unwrap_or_else(|| {
                panic!(
                    "perturbed proposedBias went unreported: {:?}",
                    report.mismatches
                )
            });
        assert_eq!(hit.left, original);
        assert_eq!(hit.right, original + 1.0);
        assert!(
            report.overlap_neurons > 0,
            "the perturbed neuron must be in the overlap"
        );
    }

    /// A missing or corrupt dump must fail loud rather than yielding an empty
    /// dump that a later diff would read as parity.
    #[test]
    fn load_compare_dump_fails_loudly_on_bad_input() {
        let dir = tempdir().unwrap();
        assert!(
            load_compare_dump(&dir.path().join("absent.json")).is_err(),
            "a missing dump must be an error"
        );

        let truncated = dir.path().join("truncated.json");
        fs::write(&truncated, r#"{"version":"test","records":2"#).unwrap();
        assert!(
            load_compare_dump(&truncated).is_err(),
            "malformed JSON must be an error"
        );

        let wrong_names = dir.path().join("wrong-names.json");
        fs::write(
            &wrong_names,
            r#"{"version":"test","records":2,"mse":0.0,"learningRate":0.01,
                "neurons":[],
                "synapses":[{"from_uuid":"input-0","to_uuid":"o1","count":1.0,
                             "currentWeight":1.0,"proposedWeight":1.0}]}"#,
        )
        .unwrap();
        assert!(
            load_compare_dump(&wrong_names).is_err(),
            "snake_case synapse UUID keys must be rejected, not silently defaulted"
        );
    }

    /// Issue #33 parity guard: `CompareDump::mse` must stay on the value the
    /// crate's own fused reduction produced before it delegated to
    /// [`neat_core::mse_record`].
    ///
    /// Records `(x → t1, t2)`: `(1 → 2, 5)` predicts `(1, 2)` ⇒ `(1 + 9)/2 =
    /// 5.0`; `(2 → 0, 1)` predicts `(2, 4)` ⇒ `(4 + 9)/2 = 6.5`. Mean over the
    /// two records is `5.75`.
    #[test]
    fn run_compare_mse_matches_pre_delegation_baseline() {
        const BASELINE_MSE: f64 = 5.75;
        let dir = tempdir().unwrap();
        let creature_path = dir.path().join("creature.json");
        fs::write(&creature_path, TWO_OUTPUT).unwrap();
        let data_dir = dir.path().join("data");
        fs::create_dir_all(&data_dir).unwrap();
        let mut f = fs::File::create(data_dir.join("0.bin")).unwrap();
        for v in [1.0f32, 2.0, 5.0, 2.0, 0.0, 1.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        let dump = run_compare(
            &creature_path,
            &data_dir,
            &BackpropConfig::default(),
            None,
            1,
            &dir.path().join("out.json"),
        )
        .unwrap();

        assert_eq!(dump.records, 2);
        assert!(
            nearly_equal(dump.mse, BASELINE_MSE),
            "mse {} drifted from the pre-delegation baseline {BASELINE_MSE}",
            dump.mse
        );
        assert_eq!(
            dump.mse - BASELINE_MSE,
            0.0,
            "the reduction must stay bit-identical, not merely within tolerance"
        );
    }

    #[test]
    fn dump_round_trip_nearly_equal() {
        let creature = parse_creature_json(
            r#"{
              "input":1,"output":1,"forwardOnly":true,
              "neurons":[{"type":"output","uuid":"o1","bias":0.1,"squash":"IDENTITY"}],
              "synapses":[{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
            }"#,
        )
        .unwrap();
        let mut signal = LearningSignal::new(1, 1);
        signal.biases[0] = BiasSignal {
            count: 2.0,
            total_adjusted_bias: 0.4,
            no_change: false,
        };
        let cfg = BackpropConfig::default();
        let dump = build_compare_dump(&creature, &signal, &cfg, 0.01, 0.0, 2, "test");
        let report = diff_compare_dumps(&dump, &dump, false).unwrap();
        assert!(report.mismatches.is_empty());
        assert_eq!(dump.neurons[0].uuid, "o1");
    }
}
