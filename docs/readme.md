\# Delta Inference Engine — Documentation



This folder contains user and developer documentation for the \*\*Delta Inference Engine\*\* crate.



\*\*Quick links\*\*

\- Getting started: `getting\_started.md`

\- Usage examples: `usage.md`

\- API reference: `api.md`

\- Development \& tests: `development.md`

\- Benchmarks: `benchmarks.md`

\- Contributing: `CONTRIBUTING.md`

\- Changelog: `CHANGELOG.md`



To build Rust API docs:

```bash

cargo doc --no-deps --open

To serve the docs/ folder locally (simple static preview):



bash

python -m http.server --directory docs 8000

Code



\---



\### `docs/getting\_started.md`

```md

\# Getting started



\## Prerequisites

\- Rust toolchain (stable) and `cargo`

\- Optional: `maturin` and Python 3.8+ for Python bindings

\- Optional: `bench` tools (Criterion) for benchmarking



\## Build the crate

From the repository root:

```bash

\# debug build

cargo build



\# release build

cargo build --release



\# build with Rayon parallel feature

cargo build --release --features parallel

Run the example

bash

cargo run --example example\_kv\_delta --release --features parallel

Run tests

bash

\# run all unit tests

cargo test



\# run a single test

cargo test test\_compute\_apply\_small



\# run tests in release mode

cargo test --release

Build Python extension (optional)

bash

\# using maturin (installs into current venv)

maturin develop --release

Code



\---



\### `docs/usage.md`

```md

\# Usage



\## Rust API (quick)

```rust

use delta\_inference::{compute\_delta\_f32, apply\_delta\_f32, ReusePolicy, DeltaRuntime, StateId};



// create master and variant buffers

let master: Vec<f32> = ...;

let variant: Vec<f32> = ...;

let id = StateId { layer: 0, head: None, name: Some("kv\_512x64".into()) };



// compute delta

let delta = compute\_delta\_f32(id.clone(), \&master, \&variant, 1e-6).unwrap();



// apply delta

let reconstructed = apply\_delta\_f32(\&master, \&delta).unwrap();

assert\_eq!(reconstructed, variant);

Runtime usage

rust

let mut runtime = DeltaRuntime::new(ReusePolicy::default());

runtime.register\_master(\&id, master.clone());

let delta = runtime.compute\_delta\_for(\&id, \&variant).unwrap();

let out = runtime.apply\_delta\_for(\&id, \&delta).unwrap();

Python usage (if built with python feature)

python

import numpy as np

from delta\_bindings import compute\_delta\_f32, apply\_delta\_f32



master = np.array(\[...], dtype=np.float32)

variant = np.array(\[...], dtype=np.float32)



delta\_json = compute\_delta\_f32(0, None, "kv\_512x64", master, variant, 1e-6)

reconstructed = apply\_delta\_f32(master, delta\_json)

Code



\---



\### `docs/api.md`

```md

\# API Reference (summary)



This file is a concise summary of the public API. For full generated docs run `cargo doc`.



\## Types

\- \*\*StateId\*\*

&#x20; - `layer: usize`

&#x20; - `head: Option<usize>`

&#x20; - `name: Option<String>`



\- \*\*Patch\*\*

&#x20; - `index: usize`

&#x20; - `value: f32`



\- \*\*Delta\*\*

&#x20; - `id: StateId`

&#x20; - `original\_len: usize`

&#x20; - `patches: Vec<Patch>`



\- \*\*KvDelta\*\* / \*\*KvDeltaPatch\*\*

&#x20; - High-level grouping of patches by layer/head/row.



\- \*\*ReusePolicy\*\*

&#x20; - Policy struct controlling reuse thresholds and tolerances.



\## Functions

\- `compute\_delta\_f32(id: StateId, master: \&\[f32], variant: \&\[f32], tolerance: f32) -> Result<Delta, DeltaError>`

&#x20; - Computes sparse patches where `|master\[i] - variant\[i]| > tolerance`.



\- `apply\_delta\_f32(master: \&\[f32], delta: \&Delta) -> Result<Vec<f32>, DeltaError>`

&#x20; - Applies patches to `master` and returns reconstructed buffer.



\## Runtime

\- `DeltaRuntime`

&#x20; - `new(policy: ReusePolicy) -> DeltaRuntime`

&#x20; - `register\_master(\&mut self, id: \&StateId, data: Vec<f32>)`

&#x20; - `compute\_delta\_for(\&self, id: \&StateId, variant: \&\[f32]) -> Result<Delta>`

&#x20; - `apply\_delta\_for(\&self, id: \&StateId, delta: \&Delta) -> Result<Vec<f32>>`



\## Errors

\- `DeltaError::LengthMismatch { expected, got }`

\- `DeltaError::Internal(String)`

