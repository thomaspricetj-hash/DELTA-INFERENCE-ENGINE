// delta_inference/examples/example_kv_delta.rs
//! Rust-only example demonstrating flat KV delta compute and apply.
//! Run with: cargo run --example example_kv_delta

use delta_inference::delta_f32::{compute_delta_f32, apply_delta_f32};
use delta_inference::state_types::StateId;

fn make_kv_tensor(rows: usize, row_len: usize, step: usize) -> (Vec<f32>, Vec<f32>) {
    let len = rows * row_len;
    let mut master = vec![0.0f32; len];
    for r in 0..rows {
        for j in 0..row_len {
            master[r * row_len + j] = ((r + j) % 256) as f32;
        }
    }
    let mut variant = master.clone();
    for r in (0..rows).step_by(step) {
        let idx = r * row_len + (r % row_len);
        // use normal floating-point addition for f32
        variant[idx] = variant[idx] + 1.0;
    }
    (master, variant)
}

fn main() {
    let rows = 512;
    let row_len = 64;
    let step = 16;
    let (master, variant) = make_kv_tensor(rows, row_len, step);
    let id = StateId { layer: 0, head: None, name: Some(format!("kv_{}x{}", rows, row_len)) };

    println!("Master len {} variant len {}", master.len(), variant.len());

    let delta = compute_delta_f32(id.clone(), &master, &variant, 1e-6).expect("compute delta");
    println!("Patches found: {}", delta.patches.len());

    let reconstructed = apply_delta_f32(&master, &delta).expect("apply delta");
    assert_eq!(reconstructed, variant);
    println!("Reconstruction successful, variant matches reconstructed output.");
}

