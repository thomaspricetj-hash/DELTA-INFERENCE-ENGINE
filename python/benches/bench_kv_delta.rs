// delta_inference/benches/bench_kv_delta.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use delta_inference::delta_f32::{compute_delta_f32, apply_delta_f32, Delta};
use delta_inference::kv_delta::{KvDelta, KvDeltaPatch};
use delta_inference::state_types::StateId;

/// Simulate a KV tensor as contiguous rows: rows * row_len elements.
fn make_kv_tensor(rows: usize, row_len: usize, step: usize) -> (Vec<f32>, Vec<f32>) {
    let len = rows * row_len;
    let mut master = vec![0.0f32; len];
    for r in 0..rows {
        for j in 0..row_len {
            master[r * row_len + j] = ((r + j) % 256) as f32;
        }
    }
    let mut variant = master.clone();
    // change one element per row every `step` rows to simulate sparse updates
    for r in (0..rows).step_by(step) {
        let idx = r * row_len + (r % row_len);
        variant[idx] = variant[idx].wrapping_add(1.0);
    }
    (master, variant)
}

fn bench_kv_row_deltas(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_delta");
    let configs = [(512usize, 64usize, 16usize), (1024, 64, 32)];
    for (rows, row_len, step) in configs {
        let (master, variant) = make_kv_tensor(rows, row_len, step);
        let id = StateId { layer: 0, head: None, name: Some(format!("kv_{}x{}", rows, row_len)) };
        group.bench_with_input(BenchmarkId::from_parameter(rows), &rows, |b, &_rows| {
            b.iter(|| {
                // compute a single flat delta for the whole KV tensor
                let delta = compute_delta_f32(id.clone(), &master, &variant, 1e-6).expect("compute");
                // convert flat delta into per-row patches (simple grouping)
                let mut kv_delta = KvDelta::new(id.clone());
                let mut row_map: std::collections::HashMap<usize, Vec<_>> = std::collections::HashMap::new();
                for p in &delta.patches {
                    let row = p.index / row_len;
                    row_map.entry(row).or_default().push(p.clone());
                }
                for (row, patches) in row_map {
                    let patches = patches.into_iter().map(|p| p).collect();
                    kv_delta.patches.push(KvDeltaPatch { layer: id.layer, head: 0, row, values: patches });
                }
                // apply delta back to master (reuse apply_delta_f32)
                let _out = apply_delta_f32(&master, &delta).expect("apply");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_kv_row_deltas);
criterion_main!(benches);
