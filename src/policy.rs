// delta_inference/src/policy.rs
use serde::{Deserialize, Serialize};

/// Reuse policy controls when to reuse previous state vs recompute.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReusePolicy {
    /// Fraction of elements that must be unchanged to allow reuse.
    pub reuse_threshold: f32,
    /// Absolute maximum patches allowed to still reuse.
    pub max_patches_for_reuse: usize,
    /// Tolerance for floating point equality.
    pub tolerance: f32,
}

impl Default for ReusePolicy {
    fn default() -> Self {
        Self {
            reuse_threshold: 0.95,
            max_patches_for_reuse: 1024,
            tolerance: 1e-6,
        }
    }
}

impl ReusePolicy {
    /// Decide whether to reuse based on number of patches and total length.
    pub fn should_reuse(&self, total_len: usize, patch_count: usize) -> bool {
        if total_len == 0 { return false; }
        let unchanged_fraction = 1.0 - (patch_count as f32 / total_len as f32);
        unchanged_fraction >= self.reuse_threshold && patch_count <= self.max_patches_for_reuse
    }
}
