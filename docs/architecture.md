Architecture

This document describes the high‑level architecture of the Delta Inference Engine, the responsibilities of each component, the data flow for KV cache deltas, storage and serialization choices, performance considerations, and extension points for integration with runtimes and Python.



Overview

The Delta Inference Engine captures and applies sparse differences between a canonical master tensor and a runtime variant tensor. It is designed to be:



Compact — store only changed elements as sparse patches.



Deterministic — apply deltas to reconstruct variants exactly.



Parallelizable — compute deltas across large buffers with Rayon.



Interoperable — expose Rust API and optional PyO3 bindings for Python.



Key concepts:



StateId: identifies a tensor by layer, optional head, and optional name.



Patch: a single changed element described by index and value.



Delta: a collection of patches plus metadata about the original tensor length and StateId.



KvDelta: higher level grouping of patches by row/head for partial application.



Components

Core Library

delta\_f32



Computes sparse patches between two \&\[f32] buffers.



Two implementations: sequential and Rayon parallel (feature parallel).



Exposes compute\_delta\_f32 and apply\_delta\_f32.



kv\_delta



Converts flat patches into row/head grouped patches for KV caches.



Supports partial application per row to avoid reconstructing entire tensors.



state\_types



StateId and lightweight descriptors used as keys for runtime and storage.



policy



ReusePolicy and heuristics that decide when to reuse existing deltas, when to send full buffers, and tolerance thresholds.



runtime



DeltaRuntime orchestrates master registration, delta computation, caching, and application.



Optional Bindings

python\_bindings



PyO3 module exposing compute and apply functions to Python with NumPy interoperability.



Tooling

Examples and benches for correctness and performance measurement.



Serialization helpers for JSON, CBOR, or MessagePack.



Data Flow

Register Master

Input: StateId, Vec<f32> master buffer.



Action: runtime stores master keyed by StateId in memory or persistent store.



Capture Variant

Input: runtime produces a variant: \&\[f32] during inference for the same StateId.



Compute Delta

Call: compute\_delta\_f32(id, \&master, \&variant, tolerance)



Output: Delta { id, original\_len, patches } where each patch is { index, value }.



Parallelization: when parallel feature is enabled, the index space is partitioned and processed with Rayon; results are merged.



Serialize and Transport

Formats: JSON for debugging; CBOR or MessagePack for production.



Header: include id, original\_len, format, and patch\_count for quick validation.



Apply Delta

Call: apply\_delta\_f32(\&master, \&delta)



Result: new Vec<f32> with patches applied; verify reconstructed == variant.



Storage and Serialization

Storage Options

In memory: runtime caches masters for low latency.



Key value store: RocksDB or LMDB for persistence and recovery. Use StateId as key.



Object store: S3 for large master snapshots; store deltas separately for incremental updates.



Serialization Recommendations

Development: JSON for readability.



Production: CBOR or MessagePack for compactness and speed.



Schema: include schema\_version, id, original\_len, patches and optional metadata (timestamp, source).



Delta Compacting

Merge multiple deltas by keeping the latest value per index.



Optionally compress dense patches as full blocks rather than sparse lists.



Performance and Scaling

Hotspots

Memory bandwidth when scanning large buffers.



Allocation churn when collecting patches; preallocate or reserve capacity when possible.



Parallel overhead for small tensors; prefer sequential path for small sizes.



Tuning Guidelines

Tolerance: set to avoid recording floating point noise. Typical range 1e-6 to 1e-3.



Chunking: process very large tensors in fixed-size chunks to reduce peak memory and enable streaming serialization.



Parallel thresholds: only enable Rayon for buffers above a size threshold to amortize thread overhead.



Patch density heuristic: if patch density exceeds a threshold, prefer sending a compressed full buffer instead of sparse patches.



Benchmarks

Measure throughput in elements/sec and end-to-end latency.



Compare sequential vs parallel implementations with representative KV sizes and patch densities.



Integration and Extensibility

Runtime Integration

Provide a small runtime API for registering masters, computing deltas, applying deltas, and storing results.



Expose hooks for custom storage backends and network transports.



Python Integration

PyO3 bindings should accept NumPy arrays and return JSON or NumPy arrays for reconstructed buffers.



Keep serialization format stable across Rust and Python.



Extension Points

Quantized types: add delta\_bf16 or delta\_i8 modules for other numeric formats.



Delta merging strategies: pluggable merge policies for multi-source updates.



Transport adapters: gRPC, REST, or custom binary protocols for delta exchange.



Security: validation layer to check original\_len and patch indices before applying.



Minimal Example Pseudocode

rust

// register master

runtime.register\_master(\&id, master\_vec);



// compute delta

let delta = compute\_delta\_f32(id.clone(), \&master, \&variant, 1e-6)?;



// serialize and send

let bytes = serde\_cbor::to\_vec(\&delta)?;



// apply on receiver

let delta: Delta = serde\_cbor::from\_slice(\&bytes)?;

let reconstructed = apply\_delta\_f32(\&master, \&delta)?;

Final notes

Keep the delta format simple and verifiable.



Favor compact binary serialization for production.



Use feature flags to control parallelism and Python bindings so consumers can choose the right tradeoffs for their environment.

