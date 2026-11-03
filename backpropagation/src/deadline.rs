//! Cooperative wall-clock deadlines for the native trainer.
//!
//! A [`Deadline`] is an absolute wall-clock instant, in milliseconds since the
//! Unix epoch, held as a `u64`. It is expired when `clock() >= epoch_ms`, so
//! the instant itself already counts as expired. Checks are cooperative: the
//! trainer reads the clock only at documented granularities (see
//! [`TimeoutPhase`]) and nothing preempts a step that is already running. Once
//! a deadline has expired the trainer starts no new unbounded scan and finishes
//! with a bounded amount of work.
//!
//! [`system_epoch_ms`] converts `SystemTime::now()` to that unit: whole
//! milliseconds since `UNIX_EPOCH`. A clock set before 1970 reads `0`, so it
//! never expires a run early; a value above `u64::MAX` milliseconds saturates
//! to `u64::MAX`.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// The current wall-clock time in milliseconds since the Unix epoch.
///
/// A clock before 1970 reads `0`; a value that does not fit in `u64` saturates
/// to `u64::MAX`.
pub fn system_epoch_ms() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(elapsed) => u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
        Err(_) => 0,
    }
}

/// An absolute wall-clock instant after which cooperative work must wind down.
///
/// The clock is injected so tests can drive expiry deterministically; the
/// production constructor [`Deadline::at_epoch_ms`] reads the system clock.
pub struct Deadline<'a> {
    epoch_ms: u64,
    clock: Box<dyn Fn() -> u64 + 'a>,
}

impl Deadline<'static> {
    /// A deadline at `epoch_ms`, measured against the system clock.
    pub fn at_epoch_ms(epoch_ms: u64) -> Self {
        Self::with_clock(epoch_ms, system_epoch_ms)
    }
}

impl<'a> Deadline<'a> {
    /// A deadline at `epoch_ms`, measured against `clock` (milliseconds since
    /// the Unix epoch). Each call to [`Deadline::expired`] reads `clock` once.
    pub fn with_clock(epoch_ms: u64, clock: impl Fn() -> u64 + 'a) -> Self {
        Self {
            epoch_ms,
            clock: Box::new(clock),
        }
    }

    /// The instant at which this deadline expires, in milliseconds since the
    /// Unix epoch.
    pub fn epoch_ms(&self) -> u64 {
        self.epoch_ms
    }

    /// Whether the clock has reached the deadline (`clock() >= epoch_ms`).
    pub fn expired(&self) -> bool {
        (self.clock)() >= self.epoch_ms
    }
}

impl fmt::Debug for Deadline<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Deadline")
            .field("epoch_ms", &self.epoch_ms)
            .finish_non_exhaustive()
    }
}

/// The trainer step at which a deadline expiry was observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TimeoutPhase {
    /// Expired before any work.
    Start,
    /// The initial error evaluation.
    BaselineEvaluation,
    /// At the top of an epoch.
    EpochStart,
    /// While accumulating record learning.
    Accumulation,
    /// While measuring a candidate's error.
    CandidateEvaluation,
    /// Before a step-halving retry.
    Backtracking,
    /// Before launching an external scorer scan.
    Scoring,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clock_reading_exactly_the_deadline_is_expired() {
        assert!(Deadline::with_clock(100, || 100).expired());
    }

    #[test]
    fn a_clock_one_millisecond_short_is_not_expired() {
        assert!(!Deadline::with_clock(100, || 99).expired());
    }

    #[test]
    fn the_maximum_deadline_is_not_expired_on_the_system_clock() {
        assert!(!Deadline::at_epoch_ms(u64::MAX).expired());
    }

    #[test]
    fn a_one_millisecond_deadline_is_expired_on_the_system_clock() {
        assert!(Deadline::at_epoch_ms(1).expired());
    }

    #[test]
    fn the_deadline_reports_the_instant_it_was_built_with() {
        assert_eq!(Deadline::with_clock(123, || 0).epoch_ms(), 123);
    }
}
