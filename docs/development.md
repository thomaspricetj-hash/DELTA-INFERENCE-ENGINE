\# Development guide



\## Code style

\- Use `rustfmt` and `clippy`.

```bash

cargo fmt

cargo clippy --all-targets --all-features -- -D warnings

Feature flags

parallel — enable Rayon parallel delta computation.



python — enable PyO3 bindings for Python.



Adding tests

Unit tests live next to modules in #\[cfg(test)] blocks.



Use cargo test for unit tests.



Use Criterion for microbenchmarks (see benches/).



Benchmarks

Add Criterion benches under benches/.



Run:



bash

cargo bench

cargo bench --features parallel

CI suggestions

Build matrix: stable Rust, run cargo test, cargo build --release --features parallel, run cargo clippy.



Optionally build Python wheel with maturin on Linux/macOS runners.



Releasing

Update CHANGELOG.md.



Tag release and push.



If publishing Python wheel, use maturin build and upload to PyPI.



Code



\---



\### `docs/benchmarks.md`

```md

\# Benchmarks



We use Criterion for microbenchmarks.



\## Running benchmarks

```bash

\# baseline (sequential)

cargo bench



\# parallel (Rayon)

cargo bench --features parallel

What to measure

Throughput: elements/sec for compute\_delta\_f32 on large buffers.



Latency: single-call latency for small KV rows.



Memory: peak memory during delta extraction.



Interpreting results

Compare sequential vs parallel to determine speedup and overhead.



Tune chunk sizes and Rayon thread pool if necessary.



Code



\---



\### `docs/CONTRIBUTING.md`

```md

\# Contributing



Thanks for contributing! Please follow these guidelines.



\## How to contribute

1\. Fork the repo and create a feature branch.

2\. Run tests and linters locally.

3\. Open a PR with a clear description and changelog entry.



\## Commit style

\- Use conventional commits (feat, fix, docs, chore).

\- Keep commits small and focused.



\## Reporting issues

\- Provide reproduction steps, platform, Rust version, and minimal code to reproduce.

