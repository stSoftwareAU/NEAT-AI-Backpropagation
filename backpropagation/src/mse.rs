//! Forward-only mean squared error over a `.bin` training directory.

use neat_core::{CompiledNetwork, CreatureExport, TrainingDataConfig, TrainingDataIterator};
use std::path::Path;

/// Whole-creature mean squared error via a forward pass (no backprop).
///
/// Returns `(mean_mse, record_count)`. Each record contributes
/// `(1/outputs) Σ (target − activation)²`, then those per-record means are
/// averaged — matching NEAT-AI `Costs.MSE`.
pub fn compute_mse(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    max_records: Option<u64>,
) -> Result<(f64, u64), String> {
    let config = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = TrainingDataIterator::new(training_data, config).map_err(|e| e.to_string())?;
    let output_n = creature.output.max(1) as f64;
    let mut sum = 0.0f64;
    let mut count = 0u64;

    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if let Some(limit) = max_records
            && count >= limit
        {
            break;
        }
        let output = network.activate(&record.inputs, creature.output);
        let mut sq = 0.0f64;
        for (pred, target) in output.iter().zip(record.outputs.iter()) {
            let d = f64::from(*pred - *target);
            sq += d * d;
        }
        sum += sq / output_n;
        count += 1;
    }
    if count == 0 {
        return Err("MSE: no training records scored".into());
    }
    Ok((sum / count as f64, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::{compile_creature, parse_creature_json};
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn identity_mse_is_zero_on_perfect_pair() {
        let dir = tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join("0.bin")).unwrap();
        f.write_all(&1.0f32.to_le_bytes()).unwrap();
        f.write_all(&1.0f32.to_le_bytes()).unwrap();
        let creature = parse_creature_json(
            r#"{
              "input":1,"output":1,"forwardOnly":true,
              "neurons":[{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
              "synapses":[{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
            }"#,
        )
        .unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let (mse, n) = compute_mse(&creature, &mut network, dir.path(), Some(1)).unwrap();
        assert_eq!(n, 1);
        assert!(mse.abs() < 1e-12);
    }
}
