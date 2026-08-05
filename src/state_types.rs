// delta_inference/src/state_types.rs
use serde::{Deserialize, Serialize};

/// Generic identifier for a tracked state (layer/head/etc).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateId {
    pub layer: usize,
    pub head: Option<usize>,
    pub name: Option<String>,
}

/// A contiguous view of tensor data to be diffed.
/// For simplicity we use f32; conversions to/from bf16/half happen at the boundary.
#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    pub id: StateId,
    pub shape: Vec<usize>,
    pub data: &'a [f32],
}

impl<'a> TensorView<'a> {
    /// Number of elements in the view.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
