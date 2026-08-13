//! Mean squared error over a `.bin` training directory.
//!
//! The loss maths lives in `neat-core`; this module only maps the creature's
//! shape onto core's streaming helper and keeps the fail-loud empty-corpus
//! contract that `train` and `sweep` rely on (issue #32).

use neat_core::{CompiledNetwork, CreatureExport, mse_mean_streaming};
use std::path::Path;

/// Whole-creature mean squared error via a forward pass (no backprop).
///
/// Delegates to [`neat_core::mse_mean_streaming`], which contributes
/// `(1/outputs) Σ (target − activation)²` per record and averages those
/// per-record means — matching NEAT-AI `Costs.MSE`.
///
/// The creature's own `forward_only` flag selects the route: only a
/// forward-only creature may take core's fused path, which skips
/// `reset_state()` between records. A recurrent creature keeps stateless
/// per-record semantics.
///
/// Returns `(mean_mse, record_count)`, or an error when the corpus scores no
/// records — core reports an empty directory as `(0.0, 0)`, which callers
/// would otherwise read as a perfect score.
pub fn compute_mse(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    max_records: Option<u64>,
) -> Result<(f64, u64), String> {
    let (mse, count) = mse_mean_streaming(
        network,
        training_data,
        creature.input,
        creature.output,
        creature.forward_only,
        max_records,
    )?;
    if count == 0 {
        return Err("MSE: no training records scored".into());
    }
    Ok((mse, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nearly_equal;
    use neat_core::{compile_creature, parse_creature_json};
    use std::io::Write;
    use tempfile::TempDir;
    use tempfile::tempdir;

    /// Two-input identity creature: `act = 0.25 + in0 − 0.5·in1`.
    const TWO_INPUT_CREATURE: &str = r#"{
      "input":2,"output":1,"forwardOnly":true,
      "neurons":[{"type":"output","uuid":"o1","bias":0.25,"squash":"IDENTITY"}],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"o1","weight":1.0},
        {"fromUUID":"input-1","toUUID":"o1","weight":-0.5}
      ]
    }"#;

    /// Ten records of `[0.25·i, 0.25·i, 0.5]`, so activation `i` is
    /// `0.25 + 0.125·i` and every value is exactly representable as `f32`.
    fn ten_record_dir() -> TempDir {
        let dir = tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join("0.bin")).unwrap();
        for i in 0..10u32 {
            let x = 0.25f32 * i as f32;
            f.write_all(&x.to_le_bytes()).unwrap();
            f.write_all(&x.to_le_bytes()).unwrap();
            f.write_all(&0.5f32.to_le_bytes()).unwrap();
        }
        dir
    }

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

    /// An empty corpus must stay a loud error — core reports `(0.0, 0)`
    /// silently, and `train`/`sweep` would read that as a perfect score.
    #[test]
    fn zero_records_is_an_error() {
        let dir = tempdir().unwrap();
        let creature = parse_creature_json(TWO_INPUT_CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let err = compute_mse(&creature, &mut network, dir.path(), None).unwrap_err();
        assert_eq!(err, "MSE: no training records scored");
    }

    /// Numeric parity with the pre-delegation implementation on >8 records
    /// (the point where core's SIMD batching kicks in). `0.2265625` is the
    /// value the local per-record loop produced on this fixture.
    #[test]
    fn ten_record_mse_matches_pre_change_value() {
        let dir = ten_record_dir();
        let creature = parse_creature_json(TWO_INPUT_CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();

        let (mse, n) = compute_mse(&creature, &mut network, dir.path(), None).unwrap();
        assert_eq!(n, 10);
        assert!(nearly_equal(mse, 0.226_562_5), "full-corpus MSE was {mse}");

        // The record cap still bounds the scan: first four records only.
        let (capped, capped_n) = compute_mse(&creature, &mut network, dir.path(), Some(4)).unwrap();
        assert_eq!(capped_n, 4);
        assert!(nearly_equal(capped, 0.023_437_5), "capped MSE was {capped}");
    }
}
