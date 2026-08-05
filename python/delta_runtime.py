# delta_inference/python/delta_runtime.py
"""
High-level Python runtime that wraps the Rust delta bindings and provides
a simple KV-delta store interface for PyTorch models.
"""

import numpy as np
from typing import Dict, Tuple, Optional
try:
    from delta_bindings import compute_delta_f32, apply_delta_f32
except Exception:
    # When building locally without the extension, provide a clear error.
    compute_delta_f32 = None
    apply_delta_f32 = None

class DeltaRuntime:
    """
    DeltaRuntime manages masters and per-step deltas for tensor states.
    Masters are stored as numpy float32 arrays.
    """

    def __init__(self, tolerance: float = 1e-6):
        if compute_delta_f32 is None or apply_delta_f32 is None:
            raise RuntimeError("Rust bindings not available. Build with `--features python` and install the extension.")
        self.masters: Dict[str, np.ndarray] = {}
        self.tolerance = float(tolerance)

    @staticmethod
    def _state_key(layer: int, head: Optional[int], name: Optional[str]) -> str:
        return f"layer:{layer}|head:{head}|name:{name}"

    def register_master(self, layer: int, head: Optional[int], name: Optional[str], data: np.ndarray):
        assert data.dtype == np.float32
        key = self._state_key(layer, head, name)
        self.masters[key] = data.copy()

    def compute_delta(self, layer: int, head: Optional[int], name: Optional[str], variant: np.ndarray) -> str:
        key = self._state_key(layer, head, name)
        master = self.masters.get(key)
        if master is None:
            raise KeyError(f"master not registered for {key}")
        if master.shape != variant.shape:
            raise ValueError("shape mismatch between master and variant")
        # call into Rust binding; returns JSON string representing Delta
        delta_json = compute_delta_f32(layer, head, name, master, variant, float(self.tolerance))
        return delta_json

    def apply_delta(self, layer: int, head: Optional[int], name: Optional[str], delta_json: str) -> np.ndarray:
        key = self._state_key(layer, head, name)
        master = self.masters.get(key)
        if master is None:
            raise KeyError(f"master not registered for {key}")
        out = apply_delta_f32(master, delta_json)
        return out

    def update_master_with_delta(self, layer: int, head: Optional[int], name: Optional[str], delta_json: str):
        key = self._state_key(layer, head, name)
        out = self.apply_delta(layer, head, name, delta_json)
        self.masters[key] = out
