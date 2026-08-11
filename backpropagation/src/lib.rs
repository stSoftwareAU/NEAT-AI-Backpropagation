//! Experimental standalone backpropagation trainer for production NEAT-AI creatures.
//!
//! The reverse-topological loop lives in `neat-core`. This crate ports the
//! Lamarck / TypeScript configuration and apply surface, then adds a
//! `trainDir`-style epoch loop and a production parity dump.

#![warn(missing_docs)]

pub mod backprop;
pub mod compare;
pub mod mse;
pub mod propagate_layout;
pub mod scorer;
pub mod sweep;
pub mod train;

pub use backprop::{
    ApplyDeltaCounts, ApplyOptions, BackpropConfig, BiasSignal, FLOAT_ABS_TOL, FLOAT_REL_TOL,
    LearningRateStrategy, LearningSignal, WeightSignal, apply_learnings, apply_learnings_with,
    calculate_learning_rate, count_apply_deltas, nearly_equal,
};
pub use compare::{
    CompareDiffReport, CompareDump, NeuronCompare, SynapseCompare, build_compare_dump,
    diff_compare_dumps,
};
pub use mse::compute_mse;
pub use propagate_layout::{
    AccumulateReport, PropagateLayout, accumulate_creature_learning,
    accumulate_creature_learning_report,
};
pub use scorer::{ScoreResult, score_creature};
pub use sweep::{SweepRequest, SweepRow, SweepSummary, run_sweep};
pub use train::{TrainJournalHeader, TrainResult, run_train};
