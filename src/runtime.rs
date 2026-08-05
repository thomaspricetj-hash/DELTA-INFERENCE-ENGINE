// delta_inference/src/runtime.rs
use crate::delta_f32::{compute_delta_f32, apply_delta_f32, Delta};
use crate::policy::ReusePolicy;
use crate::state_types::StateId;
use anyhow::Result;
use std::collections::HashMap;

/// Lightweight runtime orchestrator for delta compute/apply.
pub struct DeltaRuntime {
    /// Master states keyed by StateId (serialized key).
    pub masters: HashMap<String, Vec<f32>>,
    pub policy: ReusePolicy,
}

impl DeltaRuntime {
    /// Create a new runtime with default policy.
    pub fn new(policy: ReusePolicy) -> Self {
        Self { masters: HashMap::new(), policy }
    }

    /// Register or replace a master state.
    pub fn register_master(&mut self, id: &StateId, data: Vec<f32>) {
        self.masters.insert(serde_json::to_string(id).unwrap_or_default(), data);
    }

    /// Compute delta for a given variant buffer against the registered master.
    pub fn compute_delta_for(&self, id: &StateId, variant: &[f32]) -> Result<Delta> {
        let key = serde_json::to_string(id)?;
        let master = self.masters.get(&key)
            .ok_or_else(|| anyhow::anyhow!("master not found for id {:?}", id))?;
        let delta = compute_delta_f32(id.clone(), master.as_slice(), variant, self.policy.tolerance)?;
        Ok(delta)
    }

    /// Apply a delta to the registered master and return the reconstructed buffer.
    pub fn apply_delta_for(&self, id: &StateId, delta: &Delta) -> Result<Vec<f32>> {
        let key = serde_json::to_string(id)?;
        let master = self.masters.get(&key)
            .ok_or_else(|| anyhow::anyhow!("master not found for id {:?}", id))?;
        let out = apply_delta_f32(master.as_slice(), delta)?;
        Ok(out)
    }
}
