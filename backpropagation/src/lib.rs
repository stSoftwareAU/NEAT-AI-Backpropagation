//! Experimental standalone backpropagation trainer for production NEAT-AI creatures.
//!
//! The reverse-topological loop lives in `neat-core`. This crate ports the
//! Lamarck / TypeScript configuration and apply surface, then adds a
//! `trainDir`-style epoch loop and a production parity dump.

#![warn(missing_docs)]

pub mod acceptance;
pub mod backprop;
pub mod compare;
pub mod creature_io;
pub mod ffi;
pub mod gradient_check;
pub mod mse;
pub mod propagate_layout;
pub mod sampling;
pub mod scorer;
pub mod sweep;
pub mod tags;
pub mod trace;
pub mod train;
pub mod validate;

pub use backprop::{
    ApplyDeltaCounts, ApplyOptions, BackpropConfig, BiasSignal, FLOAT_ABS_TOL, FLOAT_REL_TOL,
    LearningRateStrategy, LearningSignal, WeightSignal, apply_learnings, apply_learnings_with,
    calculate_learning_rate, count_apply_deltas, nearly_equal,
};
pub use compare::{
    CompareDiffReport, CompareDump, NeuronCompare, SynapseCompare, build_compare_dump,
    diff_compare_dumps,
};
pub use creature_io::{
    FORWARD_ONLY_REQUIRED, ObservationWidth, check_observation_width, load_forward_only_creature,
    parse_forward_only_creature,
};
pub use ffi::{
    AbiAcceptanceMode, AbiLearningRateStrategy, NEAT_BACKPROP_ABI_VERSION,
    NEAT_BACKPROP_ERR_INVALID_ARGUMENT, NEAT_BACKPROP_ERR_PANIC, NEAT_BACKPROP_ERR_TRAIN_FAILED,
    NEAT_BACKPROP_OK, NeatBackpropBuffer, TrainAbiRequest, TrainAbiResponse,
    neat_backprop_abi_version, neat_backprop_buffer_free, neat_backprop_train,
    neat_backprop_version, train_from_json,
};
pub use gradient_check::{
    ClassStats, GeneClass, GeneProbeRow, GradientCheckRequest, GradientCheckSummary,
    run_gradient_check,
};
pub use mse::{compute_mse, compute_mse_selected};
pub use propagate_layout::{
    AccumulateReport, NeuronTraceStats, PropagateLayout, accumulate_creature_learning,
    accumulate_creature_learning_report, accumulate_creature_learning_selected,
};
pub use sampling::{
    FileSample, RecordCursor, RecordSample, RecordSelection, plan_record_sample,
    select_file_sample_indexes,
};
pub use scorer::{ScoreResult, score_creature};
pub use sweep::{SweepRequest, SweepRow, SweepSummary, run_sweep};
pub use trace::{NeuronTraceState, SynapseTraceState, build_creature_trace, write_creature_trace};
pub use train::{
    AcceptReason, AcceptanceMode, BEST_TRACE_FILE, DEFAULT_MIN_SCORE_IMPROVEMENT,
    DEFAULT_STEP_SCALE, FAILED_TRACE_DIR, ScorerAcceptance, TrainCandidateRecord, TrainCreature,
    TrainEpochRecord, TrainJournalHeader, TrainRequest, TrainResult, resolve_acceptance, run_train,
};
pub use validate::TrainedTopology;
