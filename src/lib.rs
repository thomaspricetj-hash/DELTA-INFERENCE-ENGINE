// delta_inference/src/lib.rs
#![allow(missing_docs)]

//! Delta Inference Engine core crate.
//! Exposes delta compute/apply primitives and a runtime orchestrator.
//! Feature flags:
//! - "python" enables PyO3 bindings
//! - "parallel" enables Rayon-backed parallel delta computation

pub mod state_types;
pub mod delta_f32;
pub mod kv_delta;
pub mod policy;
pub mod runtime;

pub use crate::delta_f32::{compute_delta_f32, apply_delta_f32};
pub use crate::kv_delta::{KvDelta, KvDeltaPatch};
pub use crate::policy::ReusePolicy;
pub use crate::runtime::DeltaRuntime;

#[cfg(feature = "python")]
pub mod python_bindings;
