//! The scorer-guided step-scale ladder (issue #106).
//!
//! The backtracking line search (#38) assumes slice MSE is a monotonic proxy
//! for the real scorer along the backprop direction: it starts at one step
//! scale and halves until the judge is satisfied, then stops at the *first*
//! candidate that passes. On a highly evolved production creature that
//! assumption is questionable — the scorer optimum may sit at a much smaller
//! step, or at a step whose slice MSE is slightly worse.
//!
//! A ladder replaces "halve until something passes" with "apply the one
//! accumulated learning at every configured step scale, score them all, keep
//! the best". This module owns the grid — its default and the validation that
//! refuses a rung the applier would silently rewrite — and the crate-internal
//! per-epoch evaluation (`run_ladder_epoch`) that turns it into a winner.

use crate::acceptance::{AcceptReason, ScorerAcceptance};
use crate::backprop::{
    ApplyDeltaCounts, ApplyOptions, BackpropConfig, LearningSignal, apply_learnings_with,
    count_apply_deltas,
};
use crate::creature_io::ObservationWidth;
use crate::mse::compute_mse_selected;
use crate::sampling::RecordSelection;
use crate::scorer::{ScoreResult, score_creatures};
use crate::train::TrainCandidateRecord;
use neat_core::{CreatureExport, compile_creature};
use std::path::Path;

/// Default step-scale grid for `--step-scale-ladder` (issue #106).
///
/// Seven rungs spanning two orders of magnitude below [`crate::train::DEFAULT_STEP_SCALE`],
/// which is the ladder's top rung — the grid brackets the current default
/// rather than starting from it.
pub const DEFAULT_STEP_SCALE_LADDER: [f64; 7] =
    [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01];

/// [`DEFAULT_STEP_SCALE_LADDER`] as the CLI spells it, for
/// `--step-scale-ladder` used without a value.
///
/// `default_missing_value` needs a `&'static str`, so the grid has two
/// spellings; `default_ladder_csv_parses_to_the_default_grid` pins them
/// together so neither can drift.
pub const DEFAULT_STEP_SCALE_LADDER_CSV: &str = "0.0001,0.00025,0.0005,0.001,0.0025,0.005,0.01";

/// Refuse a rung [`crate::backprop::apply_learnings_with`] would not apply as written.
///
/// `effective_step_scale` silently rewrites an unusable step scale to `1.0` and
/// caps anything above `1.0`, so a `0`, a negative, a `NaN` or a `2.0` rung
/// would be journalled as itself while a full step was applied. The journal is
/// the audit trail for a production run, so the mismatch is refused up front
/// instead of being written down wrong.
///
/// An empty ladder is refused too: a caller that asked for the ladder and
/// supplied no rungs gets an error, never a silent fall back to the line
/// search.
pub fn validate_step_scale_ladder(ladder: &[f64]) -> Result<(), String> {
    if ladder.is_empty() {
        return Err("step-scale ladder is empty — supply at least one step scale".into());
    }
    for &rung in ladder {
        if !rung.is_finite() || rung <= 0.0 || rung > 1.0 {
            return Err(format!(
                "step-scale ladder rungs must be finite and within (0, 1]: {rung}"
            ));
        }
    }
    Ok(())
}

/// Parse a comma-separated step-scale ladder, refusing an unusable rung.
///
/// An empty field (`"0.001,,0.01"`, a trailing comma, or a whitespace-only
/// grid) is a malformed grid, not a rung to skip quietly: a remote runner that
/// built the string wrong must hear about it.
pub fn parse_step_scale_ladder(raw: &str) -> Result<Vec<f64>, String> {
    if raw.trim().is_empty() {
        return Err("step-scale ladder is empty — supply at least one step scale".into());
    }
    let mut ladder = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            return Err(format!(
                "step-scale ladder has an empty rung — check the commas in '{raw}'"
            ));
        }
        ladder.push(
            trimmed
                .parse::<f64>()
                .map_err(|e| format!("invalid step-scale ladder rung '{trimmed}': {e}"))?,
        );
    }
    validate_step_scale_ladder(&ladder)?;
    Ok(ladder)
}

/// One epoch of the ladder: everything the rungs are built and judged from.
pub(crate) struct LadderEpochRequest<'a> {
    /// 1-based epoch index, journalled on every rung.
    pub epoch: u64,
    /// The creature every rung is applied to.
    pub incumbent: &'a CreatureExport,
    /// The epoch's single accumulation — every rung reuses it.
    pub learning: &'a LearningSignal,
    /// Backprop config the proposals were computed under.
    pub config: &'a BackpropConfig,
    /// The epoch's resolved learning rate.
    pub learning_rate: f64,
    /// Apply options; each rung overrides `step_scale`.
    pub apply: ApplyOptions,
    /// The step-scale grid, already validated.
    pub ladder: &'a [f64],
    /// Source observation width every candidate is checked against.
    pub width: ObservationWidth,
    /// Corpus directory for the diagnostic MSE pass.
    pub training_data: &'a Path,
    /// Records the run scores, fixed for the whole run.
    pub selection: RecordSelection<'a>,
    /// Incumbent slice MSE — the diagnostic each rung is reported against.
    pub incumbent_mse: f64,
    /// `rust_scorer` binary.
    pub scorer: &'a Path,
    /// Directory the batch scorer works in.
    pub score_dir: &'a Path,
    /// Scorer-guided settings (epsilon, optional pre-screen).
    pub settings: ScorerAcceptance,
    /// Incumbent fitness every rung is judged against.
    pub incumbent_fitness: f64,
}

/// What one ladder epoch decided.
pub(crate) struct LadderEpochOutcome {
    /// The candidate the epoch reports — the winner, or the closest rung when
    /// no rung won.
    pub candidate: CreatureExport,
    /// Gene movement counts for [`Self::candidate`].
    pub deltas: ApplyDeltaCounts,
    /// Slice MSE of [`Self::candidate`].
    pub after_mse: f64,
    /// Scorer result for [`Self::candidate`], absent when it was never scored.
    pub score: Option<ScoreResult>,
    /// Verdict for [`Self::candidate`].
    pub reason: AcceptReason,
    /// Step scale [`Self::candidate`] was applied at.
    pub step_scale: f64,
    /// Rungs the epoch evaluated.
    pub rungs: u32,
    /// One journal line per rung, in ladder order.
    pub journal: Vec<TrainCandidateRecord>,
}

/// One rung mid-evaluation.
struct Rung {
    /// Position in the ladder — the journalled `attempt`.
    index: usize,
    /// Step scale this rung applied.
    step_scale: f64,
    /// Candidate serialisation handed to the scorer.
    json: String,
    /// Slice MSE of the candidate.
    mse: f64,
    /// Scorer fitness, absent when the pre-screen dropped the rung.
    score: Option<ScoreResult>,
    /// Whether the MSE pre-screen dropped this rung before scoring.
    screened_out: bool,
}

/// Apply one accumulated learning at every rung, score the survivors in one
/// batch, and keep the best scorer improvement (issue #106).
///
/// The accumulation is the expensive part and the caller already paid for it,
/// so a rung costs an apply, a slice-MSE pass and a share of one `rust_scorer`
/// invocation. Nothing is accepted here — the outcome carries the verdict and
/// the caller decides what to do with it.
pub(crate) fn run_ladder_epoch(req: LadderEpochRequest<'_>) -> Result<LadderEpochOutcome, String> {
    validate_step_scale_ladder(req.ladder)?;
    let apply_at = |step_scale: f64| {
        apply_learnings_with(
            req.incumbent,
            req.learning,
            req.config,
            req.learning_rate,
            ApplyOptions {
                step_scale,
                ..req.apply
            },
        )
    };

    let mut rungs = Vec::with_capacity(req.ladder.len());
    for (index, &step_scale) in req.ladder.iter().enumerate() {
        let candidate = apply_at(step_scale);
        let mut net = compile_creature(&candidate).map_err(|e| e.to_string())?;
        let (mse, _) =
            compute_mse_selected(&candidate, &mut net, req.training_data, req.selection)?;
        // The pre-screen is the issue's "optional catastrophic rejection": off
        // by default, MSE is only a diagnostic and every rung is scored.
        // "Did not improve" is the negation of the line search's own `<` test,
        // so a diverged `NaN` MSE is screened out here exactly as it is there —
        // every comparison against `NaN` is false.
        let mse_improved = mse < req.incumbent_mse;
        let screened_out = req.settings.mse_pre_screen && !mse_improved;
        rungs.push(Rung {
            index,
            step_scale,
            json: req.width.checked_json_pretty(&candidate)?,
            mse,
            score: None,
            screened_out,
        });
    }

    // One scorer invocation for the whole ladder: `rust_scorer` scores a
    // directory of creatures, so the marginal cost of a rung is a file rather
    // than a process launch and another corpus pass.
    let batch: Vec<(String, &str)> = rungs
        .iter()
        .filter(|rung| !rung.screened_out)
        .map(|rung| (format!("rung-{}", rung.index), rung.json.as_str()))
        .collect();
    if !batch.is_empty() {
        let request: Vec<(&str, &str)> = batch
            .iter()
            .map(|(stem, json)| (stem.as_str(), *json))
            .collect();
        let scored = score_creatures(req.scorer, &request, req.training_data, req.score_dir)?;
        // Checked, not assumed: `zip` truncates silently, and a short result set
        // would leave a rung unscored — which `rung_reason` would then journal
        // as an MSE pre-screen rejection that never happened.
        if scored.len() != request.len() {
            return Err(format!(
                "scorer returned {} score(s) for {} ladder candidate(s)",
                scored.len(),
                request.len()
            ));
        }
        for (rung, score) in rungs
            .iter_mut()
            .filter(|rung| !rung.screened_out)
            .zip(scored)
        {
            rung.score = Some(score);
        }
    }

    // The winner is the highest fitness, not the first rung that clears the
    // epsilon — that is the whole point of scoring a ladder instead of
    // stopping at the first improver. Ties keep the smaller step: the ladder
    // is ascending, so `>` leaves the earlier rung in place.
    let winner = rungs
        .iter()
        .filter_map(|rung| rung.score.as_ref().map(|score| (rung.index, score.score)))
        .reduce(|best, next| if next.1 > best.1 { next } else { best })
        .map(|(index, _)| index);
    // Nothing scored (every rung pre-screened out) still reports a candidate:
    // the lowest-MSE rung is the one that came closest, and it is what
    // `candidate.json` and the trace store capture.
    let reported = match winner {
        Some(index) => index,
        None => rungs
            .iter()
            .reduce(|best, next| if next.mse < best.mse { next } else { best })
            .map(|rung| rung.index)
            .ok_or("step-scale ladder produced no candidates")?,
    };

    let journal: Vec<TrainCandidateRecord> = rungs
        .iter()
        .map(|rung| {
            let reason = rung_reason(rung, winner == Some(rung.index), &req);
            TrainCandidateRecord {
                kind: "candidate".into(),
                epoch: req.epoch,
                attempt: rung.index as u32,
                step_scale: rung.step_scale,
                learning_rate: req.learning_rate,
                incumbent_mse: req.incumbent_mse,
                candidate_mse: rung.mse,
                mse_delta: rung.mse - req.incumbent_mse,
                baseline_score: req.incumbent_fitness,
                candidate_score: rung.score.as_ref().map(|s| s.score),
                score_delta: rung.score.as_ref().map(|s| s.score - req.incumbent_fitness),
                accepted: reason.accepted(),
                accept_reason: reason,
            }
        })
        .collect();

    let rung = &rungs[reported];
    let reason = journal[reported].accept_reason;
    // Rebuilt rather than retained: `apply_learnings_with` is a pure function
    // of the incumbent, the learning and the step scale, so re-applying the
    // winning rung reproduces the exact candidate that was scored without
    // holding every rung's creature in memory at once.
    let candidate = apply_at(rung.step_scale);
    Ok(LadderEpochOutcome {
        deltas: count_apply_deltas(req.incumbent, &candidate, req.config.plank_constant),
        candidate,
        after_mse: rung.mse,
        score: rung.score.clone(),
        reason,
        step_scale: rung.step_scale,
        rungs: rungs.len() as u32,
        journal,
    })
}

/// Verdict for one rung: the winner is judged against the epsilon — by the same
/// [`ScorerAcceptance::verdict`] the line search applies — a scored rung that
/// lost carries [`AcceptReason::ScoreNotBest`], and an unscored rung was
/// dropped by the pre-screen.
fn rung_reason(rung: &Rung, is_winner: bool, req: &LadderEpochRequest<'_>) -> AcceptReason {
    match rung.score.as_ref() {
        None => AcceptReason::MsePreScreenRejected,
        Some(_) if !is_winner => AcceptReason::ScoreNotBest,
        Some(score) => req.settings.verdict(score.score, req.incumbent_fitness),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::DEFAULT_STEP_SCALE;

    #[test]
    fn default_ladder_csv_parses_to_the_default_grid() {
        assert_eq!(
            parse_step_scale_ladder(DEFAULT_STEP_SCALE_LADDER_CSV).unwrap(),
            DEFAULT_STEP_SCALE_LADDER.to_vec()
        );
    }

    /// The grid brackets the historical single step rather than starting above
    /// it — its top rung is exactly today's `--step-scale` default.
    #[test]
    fn the_default_ladder_tops_out_at_the_default_step_scale() {
        let top = DEFAULT_STEP_SCALE_LADDER
            .iter()
            .copied()
            .fold(f64::MIN, f64::max);
        assert!((top - DEFAULT_STEP_SCALE).abs() < 1e-12);
        assert!(DEFAULT_STEP_SCALE_LADDER.windows(2).all(|w| w[0] < w[1]));
    }

    /// Every rung the applier would rewrite is refused, so a journalled step
    /// scale is always the step scale that was applied.
    #[test]
    fn a_rung_the_applier_would_rewrite_is_refused() {
        for bad in [0.0, -0.5, f64::NAN, f64::INFINITY, 1.5] {
            let err = validate_step_scale_ladder(&[0.001, bad]).unwrap_err();
            assert!(err.contains("step-scale ladder rungs"), "{bad}: {err}");
            assert!(
                parse_step_scale_ladder(&format!("0.001,{bad}")).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn an_empty_ladder_is_refused_rather_than_ignored() {
        assert!(
            validate_step_scale_ladder(&[])
                .unwrap_err()
                .contains("empty")
        );
        for blank in ["", "   "] {
            assert!(
                parse_step_scale_ladder(blank)
                    .unwrap_err()
                    .contains("step-scale ladder is empty"),
                "{blank:?}"
            );
        }
    }

    /// A malformed grid is a fault, not a rung to skip: a stray or trailing
    /// comma from a remote runner must surface rather than silently shrink the
    /// ladder.
    #[test]
    fn an_empty_rung_is_refused_rather_than_skipped() {
        for malformed in ["0.001,,0.01", "0.001,", ",0.001", " , "] {
            let err = parse_step_scale_ladder(malformed).unwrap_err();
            assert!(err.contains("empty rung"), "{malformed:?}: {err}");
        }
    }

    #[test]
    fn a_single_rung_ladder_is_valid() {
        assert_eq!(parse_step_scale_ladder(" 1.0 ").unwrap(), vec![1.0]);
    }
}
