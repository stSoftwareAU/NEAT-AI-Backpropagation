//! Mean squared error over a `.bin` training directory.
//!
//! The loss maths lives in `neat-core`; this module only maps the creature's
//! shape onto core's streaming helper and keeps the fail-loud empty-corpus
//! contract that `train` and `sweep` rely on (issue #32).

use crate::deadline::Deadline;
use crate::sampling::{RecordCursor, RecordSelection};
use neat_core::{
    CompiledNetwork, CreatureExport, TrainingDataConfig, TrainingRecord, mse_mean_streaming,
    mse_sum_batch_packed,
};
use std::path::Path;

/// Records packed per call into core's fused batch helper on the sampled path.
///
/// Large enough for core's 8-way SIMD tier to dominate, small enough that the
/// packed buffer stays cache-friendly on a wide creature.
const SAMPLED_BATCH_RECORDS: usize = 1024;

/// Records scanned between deadline checks in a bounded error evaluation
/// ([`compute_mse_until`]).
///
/// This is the record granularity at which a bounded error evaluation checks
/// the deadline: after a batch of this many records is filled, the clock is
/// read once before the batch is processed.
pub(crate) const DEADLINE_CHECK_RECORDS: usize = 32;

/// The error of a creature over a record selection, and how much of the
/// selection it covers.
///
/// Only [`ErrorMeasure::Full`] is a complete measurement. A
/// [`ErrorMeasure::Partial`] is the mean over only the first `records` selected
/// records, so it is NOT comparable to a `Full` figure. [`ErrorMeasure::Unmeasured`]
/// means no record was scored before the scan was interrupted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorMeasure {
    /// No record was scored: the deadline expired before the first batch.
    Unmeasured,
    /// The scan was interrupted after scoring `records` records; `mse` is the
    /// mean over only those records.
    Partial {
        /// Mean squared error over the `records` processed records.
        mse: f64,
        /// Records scored before the deadline expired.
        records: u64,
    },
    /// The whole selection was scored.
    Full {
        /// Mean squared error over every selected record.
        mse: f64,
        /// Records scored.
        records: u64,
    },
}

impl ErrorMeasure {
    /// The mean squared error, but only for a [`ErrorMeasure::Full`] scan.
    pub fn full_mse(&self) -> Option<f64> {
        match self {
            Self::Full { mse, .. } => Some(*mse),
            Self::Unmeasured | Self::Partial { .. } => None,
        }
    }

    /// The mean squared error of whatever was scored, `Partial` or `Full`.
    /// `None` when nothing was scored.
    pub fn mse(&self) -> Option<f64> {
        match self {
            Self::Unmeasured => None,
            Self::Partial { mse, .. } | Self::Full { mse, .. } => Some(*mse),
        }
    }

    /// Whether the whole selection was scored.
    pub fn is_full(&self) -> bool {
        matches!(self, Self::Full { .. })
    }
}

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
    compute_mse_selected(
        creature,
        network,
        training_data,
        RecordSelection::Prefix(max_records),
    )
}

/// [`compute_mse`] over an explicit [`RecordSelection`].
///
/// A [`RecordSelection::Prefix`] takes core's streaming route unchanged. A
/// [`RecordSelection::Sample`] seeks to the sampled indexes and feeds them to
/// core's fused batch helper in blocks, so the sampled surface reports the
/// same quantity as the streaming one (issue #77).
pub fn compute_mse_selected(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    selection: RecordSelection<'_>,
) -> Result<(f64, u64), String> {
    let (mse, count) = match selection {
        RecordSelection::Prefix(max_records) => mse_mean_streaming(
            network,
            training_data,
            creature.input,
            creature.output,
            creature.forward_only,
            max_records,
        )?,
        RecordSelection::Sample(_) => sampled_mse(creature, network, training_data, selection)?,
    };
    if count == 0 {
        return Err("MSE: no training records scored".into());
    }
    Ok((mse, count))
}

/// Error of a creature over a record selection, scored in batches of
/// `batch_records` through core's fused batch helper.
///
/// `limit` caps the number of records fed in (`None` reads the whole
/// selection). `deadline`, when set, is read once per non-empty batch before
/// that batch is processed; an expired clock stops the scan without processing
/// the batch, and the result is marked incomplete. The scan is complete when
/// the selection (or `limit`) is drained, and no clock is read after the final
/// batch.
fn scan_batched(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    selection: RecordSelection<'_>,
    batch_records: usize,
    limit: Option<u64>,
    deadline: Option<&Deadline<'_>>,
) -> Result<BatchedScan, String> {
    let config = TrainingDataConfig::new(creature.input, creature.output);
    let values_per_record = config.values_per_record();
    let mut cursor = RecordCursor::open(training_data, config, selection)?;
    let mut record = TrainingRecord {
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    let mut packed: Vec<f32> = Vec::with_capacity(batch_records * values_per_record);
    let mut sum_error = 0.0f64;
    let mut count = 0u64;

    loop {
        packed.clear();
        let mut filled = 0usize;
        while filled < batch_records
            && limit.is_none_or(|max| count + (filled as u64) < max)
            && cursor.next_into(&mut record)?
        {
            packed.extend_from_slice(&record.inputs);
            packed.extend_from_slice(&record.outputs);
            filled += 1;
        }
        if filled == 0 {
            return Ok(BatchedScan {
                sum_error,
                count,
                complete: true,
            });
        }
        if deadline.is_some_and(Deadline::expired) {
            return Ok(BatchedScan {
                sum_error,
                count,
                complete: false,
            });
        }
        sum_error += mse_sum_batch_packed(
            network,
            &packed,
            creature.input,
            creature.output,
            creature.forward_only,
        );
        count += filled as u64;
    }
}

/// The outcome of one [`scan_batched`] pass.
struct BatchedScan {
    /// Sum of the per-record mean squared errors over the processed records.
    sum_error: f64,
    /// Records processed.
    count: u64,
    /// `false` when the deadline stopped the scan before the selection drained.
    complete: bool,
}

/// Score a planned sample by packing its records into core's batch helper.
fn sampled_mse(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    selection: RecordSelection<'_>,
) -> Result<(f64, u64), String> {
    let scan = scan_batched(
        creature,
        network,
        training_data,
        selection,
        SAMPLED_BATCH_RECORDS,
        None,
        None,
    )?;
    if scan.count == 0 {
        return Ok((0.0, 0));
    }
    Ok((scan.sum_error / scan.count as f64, scan.count))
}

/// [`compute_mse_selected`] that checks a deadline between record batches.
///
/// With `deadline` `None` this is exactly [`compute_mse_selected`], wrapped as
/// [`ErrorMeasure::Full`]. With `Some`, the selection is scanned in batches of
/// [`DEADLINE_CHECK_RECORDS`]; the clock is read before each non-empty batch,
/// and an expired clock stops the scan before that batch is processed.
///
/// A scan interrupted after at least one record is [`ErrorMeasure::Partial`];
/// one interrupted before any record is [`ErrorMeasure::Unmeasured`]. An
/// interrupted scan is never reported as `Full`. A corpus that scores no
/// records without expiry stays the `"MSE: no training records scored"` error.
///
/// # Errors
/// As [`compute_mse_selected`]: the selection scores no records and the
/// deadline did not interrupt the scan.
pub fn compute_mse_until(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    selection: RecordSelection<'_>,
    deadline: Option<&Deadline<'_>>,
) -> Result<ErrorMeasure, String> {
    let Some(deadline) = deadline else {
        let (mse, records) = compute_mse_selected(creature, network, training_data, selection)?;
        return Ok(ErrorMeasure::Full { mse, records });
    };
    let scan = scan_batched(
        creature,
        network,
        training_data,
        selection,
        DEADLINE_CHECK_RECORDS,
        None,
        Some(deadline),
    )?;
    if scan.complete {
        if scan.count == 0 {
            return Err("MSE: no training records scored".into());
        }
        return Ok(ErrorMeasure::Full {
            mse: scan.sum_error / scan.count as f64,
            records: scan.count,
        });
    }
    if scan.count == 0 {
        return Ok(ErrorMeasure::Unmeasured);
    }
    Ok(ErrorMeasure::Partial {
        mse: scan.sum_error / scan.count as f64,
        records: scan.count,
    })
}

/// Forward MSE over only the first `records` records of `selection`.
///
/// This is the bounded post-deadline comparison scan: it takes the same
/// batched route as [`compute_mse_selected`] but is not deadline-aware, and it
/// stops after `records` records. Returns `(mean_mse, record_count)`.
///
/// # Errors
/// The first `records` records of the selection score none, which is the same
/// `"MSE: no training records scored"` error as [`compute_mse_selected`].
pub fn compute_mse_first(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    selection: RecordSelection<'_>,
    records: u64,
) -> Result<(f64, u64), String> {
    let scan = scan_batched(
        creature,
        network,
        training_data,
        selection,
        SAMPLED_BATCH_RECORDS,
        Some(records),
        None,
    )?;
    if scan.count == 0 {
        return Err("MSE: no training records scored".into());
    }
    Ok((scan.sum_error / scan.count as f64, scan.count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nearly_equal;
    use crate::sampling::plan_record_sample;
    use neat_core::{compile_creature, parse_creature_json};
    use std::cell::Cell;
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

    /// `count` records in the [`ten_record_dir`] layout, with the input cycling
    /// through eight exactly representable values.
    fn record_dir(count: u32) -> TempDir {
        let dir = tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join("0.bin")).unwrap();
        for i in 0..count {
            let x = 0.125f32 * (i % 8) as f32;
            f.write_all(&x.to_le_bytes()).unwrap();
            f.write_all(&x.to_le_bytes()).unwrap();
            f.write_all(&0.5f32.to_le_bytes()).unwrap();
        }
        dir
    }

    #[test]
    fn until_without_a_deadline_is_compute_mse_selected() {
        let dir = record_dir(100);
        let creature = parse_creature_json(TWO_INPUT_CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();

        let (mse, records) = compute_mse_selected(
            &creature,
            &mut network,
            dir.path(),
            RecordSelection::Prefix(None),
        )
        .unwrap();
        let measured = compute_mse_until(
            &creature,
            &mut network,
            dir.path(),
            RecordSelection::Prefix(None),
            None,
        )
        .unwrap();
        assert_eq!(measured, ErrorMeasure::Full { mse, records });

        let sample = plan_record_sample(
            dir.path(),
            &TrainingDataConfig::new(creature.input, creature.output),
            100,
            1,
            true,
        )
        .unwrap();
        let (sampled_mse, sampled_n) = compute_mse_selected(
            &creature,
            &mut network,
            dir.path(),
            RecordSelection::Sample(&sample),
        )
        .unwrap();
        let sampled = compute_mse_until(
            &creature,
            &mut network,
            dir.path(),
            RecordSelection::Sample(&sample),
            None,
        )
        .unwrap();
        assert_eq!(
            sampled,
            ErrorMeasure::Full {
                mse: sampled_mse,
                records: sampled_n
            }
        );
    }

    #[test]
    fn expiry_on_the_second_check_keeps_the_first_batch() {
        let dir = record_dir(100);
        let creature = parse_creature_json(TWO_INPUT_CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();

        let reads = Cell::new(0u64);
        let deadline = Deadline::with_clock(2, || {
            reads.set(reads.get() + 1);
            reads.get()
        });
        let measured = compute_mse_until(
            &creature,
            &mut network,
            dir.path(),
            RecordSelection::Prefix(None),
            Some(&deadline),
        )
        .unwrap();
        assert_eq!(reads.get(), 2, "one check per batch until expiry");

        let ErrorMeasure::Partial { mse, records } = measured else {
            panic!("expected Partial, got {measured:?}");
        };
        assert_eq!(records, 32);
        let (capped, capped_n) =
            compute_mse(&creature, &mut network, dir.path(), Some(32)).unwrap();
        assert_eq!(capped_n, 32);
        assert!(
            nearly_equal(mse, capped),
            "partial MSE was {mse}, capped MSE was {capped}"
        );
        assert!(!measured.is_full());
        assert_eq!(measured.full_mse(), None);
        assert_eq!(measured.mse(), Some(mse));
    }

    #[test]
    fn expiry_on_the_first_check_is_unmeasured() {
        let dir = record_dir(100);
        let creature = parse_creature_json(TWO_INPUT_CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();

        let reads = Cell::new(0u64);
        let deadline = Deadline::with_clock(1, || {
            reads.set(reads.get() + 1);
            reads.get()
        });
        let measured = compute_mse_until(
            &creature,
            &mut network,
            dir.path(),
            RecordSelection::Prefix(None),
            Some(&deadline),
        )
        .unwrap();
        assert_eq!(measured, ErrorMeasure::Unmeasured);
        assert_eq!(measured.mse(), None);
        assert_eq!(reads.get(), 1);
    }

    #[test]
    fn a_never_expiring_deadline_scores_the_whole_corpus() {
        let dir = record_dir(100);
        let creature = parse_creature_json(TWO_INPUT_CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let (full, n) = compute_mse(&creature, &mut network, dir.path(), None).unwrap();

        let reads = Cell::new(0u64);
        let deadline = Deadline::with_clock(u64::MAX, || {
            reads.set(reads.get() + 1);
            0
        });
        let measured = compute_mse_until(
            &creature,
            &mut network,
            dir.path(),
            RecordSelection::Prefix(None),
            Some(&deadline),
        )
        .unwrap();
        assert!(measured.is_full());
        let mse = measured.full_mse().expect("full scan");
        assert_eq!(measured.mse(), Some(mse));
        assert_eq!(n, 100);
        assert!(
            nearly_equal(mse, full),
            "full MSE was {mse}, expected {full}"
        );
        // Four batches (32, 32, 32, 4) are each checked; the drained cursor is not.
        assert_eq!(reads.get(), 4);
    }

    #[test]
    fn sampled_scans_follow_the_same_checks() {
        let dir = record_dir(100);
        let creature = parse_creature_json(TWO_INPUT_CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        // A cap at or above the corpus size takes every record, in order.
        let sample = plan_record_sample(
            dir.path(),
            &TrainingDataConfig::new(creature.input, creature.output),
            100,
            1,
            true,
        )
        .unwrap();

        let reads = Cell::new(0u64);
        let deadline = Deadline::with_clock(2, || {
            reads.set(reads.get() + 1);
            reads.get()
        });
        let measured = compute_mse_until(
            &creature,
            &mut network,
            dir.path(),
            RecordSelection::Sample(&sample),
            Some(&deadline),
        )
        .unwrap();
        let ErrorMeasure::Partial { mse, records } = measured else {
            panic!("expected Partial, got {measured:?}");
        };
        assert_eq!(records, 32);
        let (capped, _) = compute_mse(&creature, &mut network, dir.path(), Some(32)).unwrap();
        assert!(nearly_equal(mse, capped), "partial MSE was {mse}");

        let never = Deadline::with_clock(u64::MAX, || 0);
        let full = compute_mse_until(
            &creature,
            &mut network,
            dir.path(),
            RecordSelection::Sample(&sample),
            Some(&never),
        )
        .unwrap();
        assert!(
            matches!(full, ErrorMeasure::Full { records: 100, .. }),
            "{full:?}"
        );
    }

    #[test]
    fn compute_mse_first_matches_the_capped_scan() {
        let dir = record_dir(100);
        let creature = parse_creature_json(TWO_INPUT_CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();

        let (first, first_n) = compute_mse_first(
            &creature,
            &mut network,
            dir.path(),
            RecordSelection::Prefix(None),
            10,
        )
        .unwrap();
        let (capped, capped_n) =
            compute_mse(&creature, &mut network, dir.path(), Some(10)).unwrap();
        assert_eq!(first_n, 10);
        assert_eq!(capped_n, 10);
        assert!(nearly_equal(first, capped), "first MSE was {first}");
    }

    #[test]
    fn a_zero_record_corpus_is_an_error_even_with_a_never_expiring_deadline() {
        let dir = tempdir().unwrap();
        let creature = parse_creature_json(TWO_INPUT_CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let never = Deadline::with_clock(u64::MAX, || 0);
        let err = compute_mse_until(
            &creature,
            &mut network,
            dir.path(),
            RecordSelection::Prefix(None),
            Some(&never),
        )
        .unwrap_err();
        assert_eq!(err, "MSE: no training records scored");
    }
}
