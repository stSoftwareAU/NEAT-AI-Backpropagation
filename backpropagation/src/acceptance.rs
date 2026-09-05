//! What decides whether a `train` epoch's candidate is kept (issue #104).
//!
//! Training-slice MSE is the historical judge. On an evolved production
//! creature it is a poor proxy for the objective evolution optimises, so this
//! module also carries the scorer-guided vocabulary: `NEAT-AI-scorer` fitness
//! decides, MSE becomes a journalled diagnostic, and the settings that would
//! be silently ignored on an MSE run are refused instead.

use serde::{Deserialize, Serialize};

/// Conservative default epsilon for scorer-guided acceptance (#104).
///
/// The production win protocol calls a `rust_scorer` gain a win only past
/// `1e-6`, so the trainer's own gate starts at the same margin rather than
/// keeping a candidate for a difference that is scorer noise.
pub const DEFAULT_MIN_SCORE_IMPROVEMENT: f64 = 1e-6;

/// Scorer-guided acceptance settings (#104).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScorerAcceptance {
    /// Minimum `candidate − incumbent` scorer fitness required to keep an
    /// apply. Must be finite and non-negative; defaults to
    /// [`DEFAULT_MIN_SCORE_IMPROVEMENT`].
    pub min_improvement: f64,
    /// When true, a candidate whose training-slice MSE did not fall is dropped
    /// before paying for a scorer run.
    ///
    /// Off by default, and it **is** a second gate: a candidate MSE rejects is
    /// never scored, so a scorer win MSE disagreed with is lost. Turn it on
    /// only when the scorer run is too expensive to spend on every candidate.
    pub mse_pre_screen: bool,
}

impl Default for ScorerAcceptance {
    fn default() -> Self {
        Self {
            min_improvement: DEFAULT_MIN_SCORE_IMPROVEMENT,
            mse_pre_screen: false,
        }
    }
}

impl ScorerAcceptance {
    /// Refuse an epsilon that could never gate anything.
    ///
    /// `NaN` is the dangerous one: every comparison against it is false, so an
    /// unchecked `NaN` would silently reject every candidate and read as "the
    /// scorer found nothing".
    pub fn validate(self) -> Result<Self, String> {
        if !self.min_improvement.is_finite() || self.min_improvement < 0.0 {
            return Err(format!(
                "minimum score improvement must be finite and non-negative: {}",
                self.min_improvement
            ));
        }
        Ok(self)
    }
}

/// What decides whether an epoch's candidate is kept (#104).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AcceptanceMode {
    /// Training-slice MSE — `after_mse < best_mse` (the historical default).
    #[default]
    Mse,
    /// `NEAT-AI-scorer` fitness, with MSE demoted to a diagnostic.
    Scorer(ScorerAcceptance),
}

/// Why a candidate was kept or dropped, journalled per attempt (#104).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AcceptReason {
    /// Training-slice MSE fell below the incumbent's best.
    MseImproved,
    /// Training-slice MSE did not fall.
    MseNotImproved,
    /// Kept regardless of MSE under `accept_always`.
    AcceptAlways,
    /// Scorer fitness rose by at least the configured epsilon.
    ScoreImproved,
    /// The scorer ran and the gain was below the epsilon (or negative).
    ScoreNotImproved,
    /// The scorer ran and another rung of the step-scale ladder scored higher
    /// (issue #106). The ladder keeps one winner per epoch, so a rung that
    /// improved on the incumbent but lost to a better rung is dropped under
    /// this reason rather than under [`Self::ScoreNotImproved`].
    ScoreNotBest,
    /// The optional MSE pre-screen dropped the candidate before scoring it.
    MsePreScreenRejected,
}

impl AcceptReason {
    /// Whether this verdict keeps the candidate.
    pub(crate) fn accepted(self) -> bool {
        matches!(
            self,
            Self::MseImproved | Self::AcceptAlways | Self::ScoreImproved
        )
    }
}

/// Build an [`AcceptanceMode`] from a caller's flags, refusing both an invalid
/// epsilon and scorer-guided settings a plain MSE run would silently ignore.
///
/// The CLI and the C ABI both carry `min_improvement` / `mse_pre_screen`
/// alongside the mode selector, so a caller that sets one and forgets to ask
/// for the scorer gets a run that *looks* configured. Fail loudly instead.
pub fn resolve_acceptance(
    scorer_guided: bool,
    min_improvement: f64,
    mse_pre_screen: bool,
) -> Result<AcceptanceMode, String> {
    if scorer_guided {
        return Ok(AcceptanceMode::Scorer(
            ScorerAcceptance {
                min_improvement,
                mse_pre_screen,
            }
            .validate()?,
        ));
    }
    if mse_pre_screen {
        return Err(
            "msePreScreen only applies to scorer-guided acceptance — set acceptance to \"scorer\""
                .into(),
        );
    }
    if min_improvement != DEFAULT_MIN_SCORE_IMPROVEMENT {
        return Err(format!(
            "minScoreImprovement ({min_improvement}) only applies to scorer-guided acceptance \
             — set acceptance to \"scorer\""
        ));
    }
    Ok(AcceptanceMode::Mse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_is_the_default_mode_and_the_default_epsilon_is_the_win_margin() {
        assert_eq!(AcceptanceMode::default(), AcceptanceMode::Mse);
        assert!((ScorerAcceptance::default().min_improvement - 1e-6).abs() < 1e-18);
        assert!(!ScorerAcceptance::default().mse_pre_screen);
    }

    #[test]
    fn a_non_finite_or_negative_epsilon_is_refused() {
        for bad in [f64::NAN, f64::INFINITY, -1.0] {
            let err = ScorerAcceptance {
                min_improvement: bad,
                ..ScorerAcceptance::default()
            }
            .validate()
            .unwrap_err();
            assert!(err.contains("minimum score improvement"), "{bad}: {err}");
            // The same value must be refused through the flag resolver, not
            // only through the struct.
            assert!(resolve_acceptance(true, bad, false).is_err(), "{bad}");
        }
    }

    #[test]
    fn only_score_improving_verdicts_keep_the_candidate() {
        assert!(AcceptReason::MseImproved.accepted());
        assert!(AcceptReason::AcceptAlways.accepted());
        assert!(AcceptReason::ScoreImproved.accepted());
        assert!(!AcceptReason::MseNotImproved.accepted());
        assert!(!AcceptReason::ScoreNotImproved.accepted());
        assert!(!AcceptReason::ScoreNotBest.accepted());
        assert!(!AcceptReason::MsePreScreenRejected.accepted());
    }

    #[test]
    fn modes_and_reasons_are_camel_case_on_the_wire() {
        assert_eq!(
            serde_json::to_string(&AcceptanceMode::Mse).unwrap(),
            r#""mse""#
        );
        assert_eq!(
            serde_json::to_string(&AcceptReason::MsePreScreenRejected).unwrap(),
            r#""msePreScreenRejected""#
        );
        assert_eq!(
            serde_json::to_string(&AcceptReason::ScoreNotBest).unwrap(),
            r#""scoreNotBest""#
        );
        assert_eq!(
            serde_json::to_string(&AcceptanceMode::Scorer(ScorerAcceptance::default())).unwrap(),
            r#"{"scorer":{"minImprovement":1e-6,"msePreScreen":false}}"#
        );
    }
}
