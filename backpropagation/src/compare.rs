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
