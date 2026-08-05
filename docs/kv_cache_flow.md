\# KV Cache Flow



This document describes the \*\*KV cache delta flow\*\* used by the Delta Inference Engine: how master KV state is registered, how variants are captured, how deltas are computed and applied, and recommended integration points for runtime, storage, and Python bindings.



\---



\## Overview



The KV cache flow is designed to efficiently capture sparse changes between a \*\*master\*\* KV buffer and a \*\*variant\*\* buffer produced during inference. Instead of storing full copies, the system computes a \*\*Delta\*\* (sparse patches) that records only changed elements. Deltas can be serialized, transmitted, and applied to reconstruct the variant from the master.



\*\*Goals\*\*

\- Minimize memory and bandwidth for KV updates.

\- Keep compute simple and parallelizable.

\- Provide deterministic reconstruction and compact serialization.



\---



\## High-level flow



1\. \*\*Register master\*\*  

&#x20;  - The runtime stores a canonical master buffer keyed by a `StateId` (layer/head/name).

&#x20;  - `DeltaRuntime::register\_master(\&id, data)` persists the master in memory.



2\. \*\*Capture variant\*\*  

&#x20;  - During inference, capture the current KV buffer (variant) for the same `StateId`.



3\. \*\*Compute delta\*\*  

&#x20;  - Call `compute\_delta\_f32(id, \&master, \&variant, tolerance)` to produce a `Delta`:

&#x20;    - `Delta.patches` is a list of `(index, value)` where `|master\[i] - variant\[i]| > tolerance`.

&#x20;    - Implementation supports sequential and Rayon-parallel modes (feature `parallel`).



4\. \*\*Serialize / store / transmit\*\*  

&#x20;  - Serialize `Delta` (e.g., JSON, CBOR, MessagePack) for persistence or network transfer.

&#x20;  - For large-scale systems prefer binary formats (CBOR/MessagePack) to reduce overhead.



5\. \*\*Apply delta\*\*  

&#x20;  - Reconstruct variant with `apply\_delta\_f32(\&master, \&delta)` which returns a new `Vec<f32>` with patches applied.



6\. \*\*Optional: Merge / compact\*\*  

&#x20;  - Merge multiple deltas for the same master by applying them in order or by compacting patches (keep latest value per index).

&#x20;  - Optionally convert to `KvDelta` grouping by layer/head/row for efficient partial application.



\---



\## Data model



\*\*StateId\*\*

\- \*\*layer\*\*: `usize` — model layer index.

\- \*\*head\*\*: `Option<usize>` — optional attention head.

\- \*\*name\*\*: `Option<String>` — human-readable identifier.



\*\*Patch\*\*

\- \*\*index\*\*: `usize` — flattened index into the tensor.

\- \*\*value\*\*: `f32` — new value for that index.



\*\*Delta\*\*

\- \*\*id\*\*: `StateId`

\- \*\*original\_len\*\*: `usize` — length of the master buffer.

\- \*\*patches\*\*: `Vec<Patch>`



\*\*KvDelta / KvDeltaPatch\*\* (higher-level)

\- Group patches by `layer`, `head`, and `row` for row-level KV storage and partial reconstruction.



\---



\## API hooks and examples



\### Register master

```rust

let mut runtime = DeltaRuntime::new(ReusePolicy::default());

runtime.register\_master(\&id, master\_vec);



