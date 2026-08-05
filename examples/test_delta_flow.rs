// delta_inference/examples/test_delta_flow.rs

use delta_inference_engine::{
    compute_delta_f32,
    apply_delta_f32,
    StateId,
};

fn main() {
    // --- USER SUPPLIED MODEL KV TENSORS ---
    // Replace these with KV tensors from *their* model.
    // Example: model.forward() -> kv_cache[layer][head]
    let master: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let mut variant = master.clone();
    variant[2] = 2.75; // simulate model change

    // --- IDENTIFIER FOR THIS KV BLOCK ---
    let id = StateId {
        layer: 0,
        head: Some(0),
        name: Some("test_kv".into()),
    };

    // --- COMPUTE DELTA ---
    let delta = compute_delta_f32(id.clone(), &master, &variant, 1e-6)
        .expect("delta compute failed");

    println!("PATCH COUNT: {}", delta.patches.len());
    for p in &delta.patches {
        println!("patch index={} value={}", p.index, p.value);
    }

    // --- APPLY DELTA ---
    let reconstructed = apply_delta_f32(&master, &delta)
        .expect("delta apply failed");

    // --- VERIFY ---
    let matches = reconstructed == variant;
    println!("RECONSTRUCTED MATCHES VARIANT: {}", matches);

    if !matches {
        println!("ERROR: reconstruction mismatch");
    } else {
        println!("SUCCESS: delta engine validated");
    }
}
