//! Seeded record sampling for `--max-records` (issue #77).
//!
//! `--max-records N` used to mean "the first N records in directory scan
//! order" — core truncates the file list and the accumulate loop breaks after
//! N. A capped epoch therefore over-fitted the earliest files of a corpus and
//! never saw the later ones.
//!
//! NEAT-AI's TypeScript trainer treats the cap as a *rate* instead
//! (`selectFileSampleIndexes`): per `.bin` file it shuffles the record
//! indexes, takes `ceil(fileRecords × rate)` of them, then sorts the take
//! ascending so disk reads stay sequential. This module ports that selection
//! so the Rust trainer draws the same shape of sample.
//!
//! The contract, in one line: **`--max-records N` is a rate of
//! `N / total_records` applied to every file**, so the realised count can
//! round up by at most one record per file — exactly as the TypeScript does.

use neat_core::{
    SeekingRecordReader, TrainingDataConfig, TrainingDataIterator, TrainingRecord, find_bin_files,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::path::{Path, PathBuf};

/// The records selected from one `.bin` file.
#[derive(Debug, Clone)]
pub struct FileSample {
    /// The `.bin` file the indexes address.
    pub path: PathBuf,
    /// Selected zero-based record indexes, ascending.
    pub indexes: Vec<u64>,
}

/// A whole-directory record sample.
#[derive(Debug, Clone)]
pub struct RecordSample {
    files: Vec<FileSample>,
    total_records: u64,
    rate: f64,
}

impl RecordSample {
    /// Per-file selections, in directory scan order. Files that contributed no
    /// record are omitted.
    pub fn files(&self) -> &[FileSample] {
        &self.files
    }

    /// Records this sample visits.
    pub fn selected(&self) -> u64 {
        self.files.iter().map(|f| f.indexes.len() as u64).sum()
    }

    /// Records the whole corpus holds.
    pub fn total_records(&self) -> u64 {
        self.total_records
    }

    /// Fraction of each file that was drawn.
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

/// Which records a pass must visit.
#[derive(Debug, Clone, Copy)]
pub enum RecordSelection<'a> {
    /// Sequential scan, stopping after a prefix of `n` records when set.
    Prefix(Option<u64>),
    /// A seeded per-file sample built by [`plan_record_sample`].
    Sample(&'a RecordSample),
}

/// Choose the record indexes to draw from one file.
///
/// Port of NEAT-AI `selectFileSampleIndexes`: shuffle (unless
/// `disable_random_samples`), take `ceil(file_records × rate)`, sort ascending.
/// `rate` is clamped to `0.0..=1.0`; a non-finite rate selects nothing so the
/// caller's empty-corpus guard fails loud rather than scoring a partial epoch.
pub fn select_file_sample_indexes(
    file_records: u64,
    rate: f64,
    disable_random_samples: bool,
    rng: &mut impl Rng,
) -> Vec<u64> {
    let rate = if rate.is_finite() {
        rate.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut indexes: Vec<u64> = (0..file_records).collect();
    if !disable_random_samples {
        indexes.shuffle(rng);
    }
    let take = ((file_records as f64) * rate).ceil() as u64;
    indexes.truncate(take.min(file_records) as usize);
    indexes.sort_unstable();
    indexes
}

/// Plan the sample a `--max-records` cap resolves to over a whole directory.
///
/// The cap becomes a rate of `max_records / total_records` (capped at `1.0`),
/// which is then applied per file by [`select_file_sample_indexes`]. `seed`
/// makes the draw reproducible; `disable_random_samples` reduces it to each
/// file's leading prefix.
///
/// # Errors
/// - a `.bin` file cannot be listed, stat-ed or opened, or ends mid-record;
/// - `max_records` is zero;
/// - the corpus holds no whole record.
pub fn plan_record_sample(
    training_data: &Path,
    config: &TrainingDataConfig,
    max_records: u64,
    seed: u64,
    disable_random_samples: bool,
) -> Result<RecordSample, String> {
    if max_records == 0 {
        return Err("record sampling: max records must be at least 1".into());
    }
    let paths = find_bin_files(training_data).map_err(|e| {
        format!(
            "record sampling: failed to list .bin files in '{}': {e}",
            training_data.display()
        )
    })?;

    let mut counts = Vec::with_capacity(paths.len());
    let mut total_records = 0u64;
    for path in &paths {
        // Opening through core's reader keeps the trailing-bytes check on this
        // path too — a corpus that ends mid-record fails loud here rather than
        // silently scoring a short epoch.
        let reader = SeekingRecordReader::open(path, config.clone())
            .map_err(|e| format!("record sampling: '{}': {e}", path.display()))?;
        let records = reader.total_records();
        total_records += records;
        counts.push(records);
    }
    if total_records == 0 {
        return Err(format!(
            "record sampling: no training records in '{}'",
            training_data.display()
        ));
    }

    let rate = if max_records >= total_records {
        1.0
    } else {
        max_records as f64 / total_records as f64
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut files = Vec::with_capacity(paths.len());
    for (path, file_records) in paths.into_iter().zip(counts) {
        let indexes =
            select_file_sample_indexes(file_records, rate, disable_random_samples, &mut rng);
        if indexes.is_empty() {
            continue;
        }
        files.push(FileSample { path, indexes });
    }

    Ok(RecordSample {
        files,
        total_records,
        rate,
    })
}

/// Streaming reader over the records a [`RecordSelection`] names.
///
/// One cursor type behind both surfaces (accumulate and eval MSE) is what
/// keeps an epoch's two passes on the identical record set.
pub struct RecordCursor<'a> {
    inner: CursorInner<'a>,
}

/// Backing state for the two selection routes.
enum CursorInner<'a> {
    /// Sequential directory scan with an optional prefix cap.
    Prefix {
        iter: TrainingDataIterator,
        limit: Option<u64>,
        seen: u64,
    },
    /// Seek-per-index reads over a planned sample.
    Sample {
        config: TrainingDataConfig,
        files: &'a [FileSample],
        file_pos: usize,
        index_pos: usize,
        reader: Option<SeekingRecordReader>,
    },
}

impl<'a> RecordCursor<'a> {
    /// Open a cursor over `training_data` for `selection`.
    ///
    /// `training_data` is only read for [`RecordSelection::Prefix`]; a sample
    /// already carries the file paths it was planned from.
    pub fn open(
        training_data: &Path,
        config: TrainingDataConfig,
        selection: RecordSelection<'a>,
    ) -> Result<Self, String> {
        let inner = match selection {
            RecordSelection::Prefix(limit) => CursorInner::Prefix {
                iter: TrainingDataIterator::new(training_data, config)
                    .map_err(|e| e.to_string())?,
                limit,
                seen: 0,
            },
            RecordSelection::Sample(sample) => CursorInner::Sample {
                config,
                files: sample.files(),
                file_pos: 0,
                index_pos: 0,
                reader: None,
            },
        };
        Ok(Self { inner })
    }

    /// Read the next selected record into `record`, returning `false` once the
    /// selection is exhausted.
    pub fn next_into(&mut self, record: &mut TrainingRecord) -> Result<bool, String> {
        match &mut self.inner {
            CursorInner::Prefix { iter, limit, seen } => {
                if let Some(limit) = limit
                    && *seen >= *limit
                {
                    return Ok(false);
                }
                if iter.next_record_into(record).map_err(|e| e.to_string())? {
                    *seen += 1;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            CursorInner::Sample {
                config,
                files,
                file_pos,
                index_pos,
                reader,
            } => {
                loop {
                    let Some(file) = files.get(*file_pos) else {
                        return Ok(false);
                    };
                    if reader.is_none() {
                        *reader = Some(
                            SeekingRecordReader::open(&file.path, config.clone())
                                .map_err(|e| format!("'{}': {e}", file.path.display()))?,
                        );
                        *index_pos = 0;
                    }
                    match file.indexes.get(*index_pos) {
                        Some(&index) => {
                            *index_pos += 1;
                            let open = reader
                                .as_mut()
                                .ok_or_else(|| "record sampling: reader vanished".to_string())?;
                            open.read_record_into(index, record).map_err(|e| {
                                format!(
                                    "record sampling: '{}' record {index}: {e}",
                                    file.path.display()
                                )
                            })?;
                            return Ok(true);
                        }
                        None => {
                            // File exhausted — advance and open the next one.
                            *file_pos += 1;
                            *reader = None;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    /// Deterministic RNG for the shuffled cases.
    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    #[test]
    fn disabled_random_samples_take_the_leading_prefix() {
        let mut r = rng(1);
        assert_eq!(
            select_file_sample_indexes(10, 0.35, true, &mut r),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn shuffled_indexes_are_sorted_ascending() {
        let mut r = rng(2);
        let picked = select_file_sample_indexes(100, 0.1, false, &mut r);
        assert_eq!(picked.len(), 10);
        assert!(picked.windows(2).all(|w| w[0] < w[1]));
        assert!(picked.iter().all(|i| *i < 100));
    }

    #[test]
    fn the_take_is_ceiled_and_capped_at_the_file_length() {
        let mut r = rng(3);
        // ceil(7 × 0.5) = 4, not 3.
        assert_eq!(select_file_sample_indexes(7, 0.5, true, &mut r).len(), 4);
        // A rate above 1 cannot select more records than the file holds.
        assert_eq!(select_file_sample_indexes(7, 4.0, true, &mut r).len(), 7);
        // A non-finite rate selects nothing rather than panicking.
        assert!(select_file_sample_indexes(7, f64::NAN, true, &mut r).is_empty());
        assert!(select_file_sample_indexes(0, 1.0, false, &mut r).is_empty());
    }

    #[test]
    fn a_zero_cap_is_rejected() {
        let dir = tempdir().unwrap();
        let err = plan_record_sample(dir.path(), &TrainingDataConfig::new(1, 1), 0, 1, false)
            .unwrap_err();
        assert!(err.contains("at least 1"), "{err}");
    }

    #[test]
    fn a_corpus_ending_mid_record_fails_loud() {
        let dir = tempdir().unwrap();
        // 6 bytes cannot form an 8-byte (1 input + 1 output) record.
        std::fs::write(dir.path().join("0.bin"), [0u8; 6]).unwrap();
        let err = plan_record_sample(dir.path(), &TrainingDataConfig::new(1, 1), 4, 1, false)
            .unwrap_err();
        assert!(err.contains("record sampling"), "{err}");
    }

    #[test]
    fn the_cursor_reads_exactly_the_selected_records() {
        let dir = tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join("0.bin")).unwrap();
        for i in 0..8u32 {
            f.write_all(&(i as f32).to_le_bytes()).unwrap();
            f.write_all(&(10.0 + i as f32).to_le_bytes()).unwrap();
        }
        drop(f);

        let cfg = TrainingDataConfig::new(1, 1);
        let sample = plan_record_sample(dir.path(), &cfg, 4, 1, true).unwrap();
        let mut cursor =
            RecordCursor::open(dir.path(), cfg, RecordSelection::Sample(&sample)).unwrap();
        let mut record = TrainingRecord {
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        let mut seen = Vec::new();
        while cursor.next_into(&mut record).unwrap() {
            seen.push((record.inputs[0], record.outputs[0]));
        }
        assert_eq!(
            seen,
            vec![(0.0, 10.0), (1.0, 11.0), (2.0, 12.0), (3.0, 13.0)]
        );
    }

    #[test]
    fn the_prefix_cursor_honours_its_cap() {
        let dir = tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join("0.bin")).unwrap();
        for i in 0..8u32 {
            f.write_all(&(i as f32).to_le_bytes()).unwrap();
            f.write_all(&(10.0 + i as f32).to_le_bytes()).unwrap();
        }
        drop(f);

        let cfg = TrainingDataConfig::new(1, 1);
        let mut cursor =
            RecordCursor::open(dir.path(), cfg, RecordSelection::Prefix(Some(3))).unwrap();
        let mut record = TrainingRecord {
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        let mut count = 0;
        while cursor.next_into(&mut record).unwrap() {
            count += 1;
        }
        assert_eq!(count, 3);
    }
}
