Policy Design

This document defines the policy primitives, heuristics, and operational rules that govern delta computation, reuse, transmission, and storage for the Delta Inference Engine. It is intended for runtime implementers, integrators, and system architects.



Goals and Principles

Minimize bandwidth and storage by sending only meaningful changes.



Preserve correctness: deltas must reconstruct variants deterministically.



Be adaptive: policies should tune behavior based on patch density, latency, and resource constraints.



Fail safe: when in doubt, prefer correctness and safety over aggressive optimization.



Observable and tunable: expose metrics and knobs for production tuning.



Policy Primitives

ReusePolicy

reuse\_threshold: f32 — minimum absolute difference to consider a value changed.



max\_patch\_ratio: f32 — maximum fraction of elements that can be patched before preferring full buffer transfer.



chunk\_size: usize — size in elements for chunked delta computation and streaming.



parallel\_threshold: usize — buffer length above which parallel computation is enabled.



merge\_strategy: enum { KeepLatest, KeepFirst, Overwrite } — how to merge multiple deltas for same master.



serialize\_format: enum { Json, Cbor, MsgPack } — preferred serialization format.



sparsity\_compression: enum { None, RLE, Block } — optional compression for patch lists.



max\_patch\_count\_for\_reuse: usize — absolute cap on patches allowed for reuse path.



send\_full\_when\_dense: bool — if true, send full buffer when patch density exceeds threshold.



Runtime Metrics

patch\_count: number of patches produced for a delta.



patch\_ratio: patch\_count / original\_len.



compute\_time\_ms: time to compute delta.



apply\_time\_ms: time to apply delta.



serialize\_bytes: serialized delta size.



network\_latency\_ms: measured roundtrip for delta transfer.



Heuristics and Decision Flow

Compute Delta



Use reuse\_threshold as the tolerance for element comparison.



Compute in chunks of chunk\_size to bound memory and enable streaming.



Evaluate Patch Density



Compute patch\_ratio.



If patch\_ratio > max\_patch\_ratio OR patch\_count > max\_patch\_count\_for\_reuse then prefer full buffer or a compressed block transfer.



Otherwise proceed with sparse delta path.



Serialization Choice



Use serialize\_format unless the runtime detects a better option (e.g., CBOR for binary pipelines).



If sparsity\_compression is set, compress patch list before serialization.



Parallelization



If original\_len >= parallel\_threshold and parallel feature enabled, use Rayon to compute patches in parallel.



For small buffers, use sequential path to avoid thread overhead.



Merging and Ordering



When merging deltas for the same StateId, apply merge\_strategy.



Prefer deterministic ordering: sort patches by index before serialization when merging.



Partial Application



If KvDelta grouping is available, prefer row-level deltas to allow partial application without reconstructing entire tensor.



Security and Validation Rules

Length Validation: reject deltas where delta.original\_len != master.len() before applying.



Index Bounds: validate every patch index 0 <= index < original\_len. Reject or sanitize deltas with invalid indices.



Patch Count Limits: enforce max\_patch\_count\_for\_reuse and reject deltas exceeding configured caps.



Schema Versioning: include schema\_version in serialized deltas and reject unknown versions.



Authentication: require signed or authenticated transport for deltas in multi-tenant or untrusted networks.



Rate Limiting: throttle delta application requests to protect memory and CPU under load.



Serialization and Transport Policies

Header: every serialized payload must include schema\_version, id, original\_len, patch\_count, format, and optional metadata (timestamp, source).



Streaming: for large tensors, stream deltas chunk-by-chunk with per-chunk headers to allow early rejection.



Fallback: if deserialization fails, fall back to requesting the full master snapshot.



Compression: enable compression for network transport when serialize\_bytes exceeds a configurable threshold.



Merge and Compaction Strategies

KeepLatest (default): later deltas override earlier ones for the same index. Implementation: build a hashmap index→value from deltas in chronological order, then emit sorted patches.



KeepFirst: first observed value wins; ignore later patches for the same index.



Overwrite: apply deltas in order without compaction; useful when order matters and you want to preserve history.



Compaction routine



Input: list of deltas for same StateId.



Build index→value map using merge\_strategy.



Emit compacted Delta with sorted patch indices.



Operational Tuning and Recommendations

Default thresholds



reuse\_threshold = 1e-6



max\_patch\_ratio = 0.02 (2%)



chunk\_size = 16\_384 elements



parallel\_threshold = 65\_536 elements



max\_patch\_count\_for\_reuse = 100\_000



When to increase reuse\_threshold



High floating point noise or quantized models; raising threshold reduces spurious patches.



When to lower max\_patch\_ratio



Low bandwidth environments; prefer full buffer transfer only when patches are very sparse.



When to enable sparsity\_compression



When patch lists are long but contain runs or block patterns; RLE or block encoding reduces size.



Monitoring



Track patch\_ratio and serialize\_bytes over time and adjust thresholds automatically or via operator alerts.



Testing and Validation

Unit tests



Validate compute\_delta\_f32 and apply\_delta\_f32 roundtrip for random seeds, edge cases, and tolerance boundaries.



Test merge strategies with overlapping patches.



Integration tests



End-to-end tests: register master, compute delta, serialize, deserialize, apply, and verify equality.



Fault injection: corrupted patch indices, mismatched lengths, and invalid schema versions.



Benchmarks



Measure compute throughput and latency across representative KV sizes and patch densities.



Compare sequential vs parallel implementations and tune parallel\_threshold.



Example Policy Configuration

toml

\[reuse\_policy]

reuse\_threshold = 1e-6

max\_patch\_ratio = 0.02

chunk\_size = 16384

parallel\_threshold = 65536

merge\_strategy = "KeepLatest"

serialize\_format = "Cbor"

sparsity\_compression = "RLE"

max\_patch\_count\_for\_reuse = 100000

send\_full\_when\_dense = true

Change Management

Version policy: bump schema\_version on any incompatible change to delta format.



Rollout: deploy policy changes behind feature flags and monitor metrics for regressions.



Audit: log decisions when switching from sparse delta to full buffer to aid tuning.

