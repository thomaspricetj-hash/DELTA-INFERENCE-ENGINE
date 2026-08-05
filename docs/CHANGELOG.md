\# Changelog



All notable changes to this project will be documented in this file.



\## Unreleased

\- Add parallel delta computation behind `parallel` feature.

\- Add Python bindings via PyO3 (feature `python`).

\- Example: `example\_kv\_delta` demonstrates compute/apply flow.



\## 0.1.0 - YYYY-MM-DD

\- Initial public release.

Quick commands to preview and publish docs

Open Rust API docs



bash

cargo doc --no-deps --open

Serve docs/ locally



bash

python -m http.server --directory docs 8000

\# then open http://localhost:8000

Optional: use MkDocs for nicer site



bash

pip install mkdocs mkdocs-material

\# create mkdocs.yml and run

mkdocs serve

