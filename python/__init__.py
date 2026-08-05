# delta_inference/python/__init__.py
"""
Delta-Inference Python package.

Expose:
- delta_bindings: low-level Rust-backed delta compute/apply
- delta_runtime: high-level runtime wrapper for PyTorch integration
"""
from .delta_runtime import DeltaRuntime
from .model_hooks import attach_kv_hooks

__all__ = ["DeltaRuntime", "attach_kv_hooks"]
