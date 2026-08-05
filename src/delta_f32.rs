// delta_inference/src/delta_f32.rs
use crate::state_types::StateId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A single sparse patch: index and new value.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Patch {
    pub index: usize,
    pub value: f32,
}

/// Delta representation for an entire tensor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Delta {
    pub id: StateId,
    pub original_len: usize,
    pub patches: Vec<Patch>,
}

/// Errors from delta operations.
#[derive(Debug, Error)]
pub enum DeltaError {
    #[error("length mismatch: expected {expected} got {got}")]
    LengthMismatch { expected: usize, got: usize },

    #[error("internal error: {0}")]
    Internal(String),
}

/// Sequential implementation (compiled only when the `parallel` feature is NOT enabled).
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
fn compute_delta_f32_seq(
    id: StateId,
    master: &[f32],
    variant: &[f32],
    tolerance: f32,
) -> Result<Delta, DeltaError> {
    if master.len() != variant.len() {
        return Err(DeltaError::LengthMismatch { expected: master.len(), got: variant.len() });
    }

    let mut patches = Vec::new();
    for (i, (&m, &v)) in master.iter().zip(variant.iter()).enumerate() {
        if (m - v).abs() > tolerance {
            patches.push(Patch { index: i, value: v });
        }
    }

    Ok(Delta { id, original_len: master.len(), patches })
}

/// Parallel implementation (compiled only when the `parallel` feature IS enabled).
#[cfg(feature = "parallel")]
fn compute_delta_f32_par(
    id: StateId,
    master: &[f32],
    variant: &[f32],
    tolerance: f32,
) -> Result<Delta, DeltaError> {
    use rayon::prelude::*;

    if master.len() != variant.len() {
        return Err(DeltaError::LengthMismatch { expected: master.len(), got: variant.len() });
    }

    let len = master.len();
    let patches: Vec<Patch> = (0..len)
        .into_par_iter()
        .filter_map(|i| {
            let m = master[i];
            let v = variant[i];
            if (m - v).abs() > tolerance {
                Some(Patch { index: i, value: v })
            } else {
                None
            }
        })
        .collect();

    Ok(Delta { id, original_len: len, patches })
}

/// Public API: dispatch to parallel implementation when feature enabled.
pub fn compute_delta_f32(
    id: StateId,
    master: &[f32],
    variant: &[f32],
    tolerance: f32,
) -> Result<Delta, DeltaError> {
    #[cfg(feature = "parallel")]
    {
        return compute_delta_f32_par(id, master, variant, tolerance);
    }
    #[cfg(not(feature = "parallel"))]
    {
        return compute_delta_f32_seq(id, master, variant, tolerance);
    }
}

/// Apply a delta to a master buffer producing a new buffer.
/// Returns a newly allocated Vec<f32> with patches applied.
pub fn apply_delta_f32(master: &[f32], delta: &Delta) -> Result<Vec<f32>, DeltaError> {
    if master.len() != delta.original_len {
        return Err(DeltaError::LengthMismatch { expected: master.len(), got: delta.original_len });
    }

    let mut out = master.to_vec();
    for p in &delta.patches {
        if p.index >= out.len() {
            return Err(DeltaError::Internal(format!("patch index {} out of bounds", p.index)));
        }
        out[p.index] = p.value;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_types::StateId;

    #[test]
    fn test_compute_apply_small() {
        let id = StateId { layer: 0, head: None, name: Some("t".into()) };
        let master = vec![0.0f32, 1.0, 2.0, 3.0];
        let mut variant = master.clone();
        variant[2] = 2.5;
        let delta = compute_delta_f32(id.clone(), &master, &variant, 1e-6).expect("compute");
        assert_eq!(delta.patches.len(), 1);
        let out = apply_delta_f32(&master, &delta).expect("apply");
        assert_eq!(out, variant);
    }
}

