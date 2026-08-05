// delta_inference/src/kv_delta.rs
use crate::delta_f32::Patch;
use crate::state_types::StateId;
use serde::{Deserialize, Serialize};

/// Patch metadata for KV rows (layer, head, row index).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvDeltaPatch {
    pub layer: usize,
    pub head: usize,
    pub row: usize,
    pub values: Vec<Patch>, // patches within the row
}

/// High level KV delta container for a single step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvDelta {
    pub id: StateId,
    pub patches: Vec<KvDeltaPatch>,
}

impl KvDelta {
    /// Create an empty KV delta.
    pub fn new(id: StateId) -> Self {
        Self { id, patches: Vec::new() }
    }

    /// Add a row patch.
    pub fn add_row_patch(&mut self, layer: usize, head: usize, row: usize, values: Vec<Patch>) {
        self.patches.push(KvDeltaPatch { layer, head, row, values });
    }
}

