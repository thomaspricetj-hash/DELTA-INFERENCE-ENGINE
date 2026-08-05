# delta_inference/examples/example_pytorch_integration.py
"""
Minimal PyTorch integration example for Delta-Inference Engine.

This example:
- creates a tiny transformer-like module that emits KV tensors
- attaches hooks using attach_kv_hooks
- registers a master KV in DeltaRuntime
- computes a delta after a single forward step
- applies the delta and updates the master

Requirements:
- PyTorch installed
- delta_inference Python package built and installed (maturin develop or pip wheel)
"""

import torch
import numpy as np
from delta_inference import DeltaRuntime, attach_kv_hooks

# Tiny transformer-like module that exposes a KV tensor in forward
class TinyKVModule(torch.nn.Module):
    def __init__(self, embed_dim=32, seq_len=8):
        super().__init__()
        self.embed = torch.nn.Embedding(256, embed_dim)
        self.linear_k = torch.nn.Linear(embed_dim, embed_dim)
        self.linear_v = torch.nn.Linear(embed_dim, embed_dim)
        self.seq_len = seq_len

    def forward(self, tokens):
        # tokens: (batch, seq)
        x = self.embed(tokens)                       # (B, S, D)
        k = self.linear_k(x)                         # (B, S, D)
        v = self.linear_v(x)                         # (B, S, D)
        # For demo we return k and v so hooks can capture them
        return {"k": k, "v": v, "out": x.mean(dim=-1)}

def main():
    # Build model and runtime
    model = TinyKVModule(embed_dim=64, seq_len=8)
    rt = DeltaRuntime(tolerance=1e-6)

    # Simple capture function that registers master on first capture and computes delta on second
    capture_state = {}

    def capture_fn(layer_name, head, name, tensor):
        # tensor is a torch.Tensor on CPU or GPU; convert to CPU float32 numpy
        t = tensor.detach().cpu()
        if t.dtype != torch.float32:
            t = t.float()
        arr = t.numpy()
        # Flatten to 1D for delta engine
        flat = arr.ravel().astype(np.float32)
        key = f"{layer_name}|{name}"
        if key not in capture_state:
            # register master
            print(f"Registering master for {key} shape={arr.shape}")
            rt.register_master(0, None, key, flat)
            capture_state[key] = "master_registered"
        else:
            # compute delta against registered master
            print(f"Computing delta for {key}")
            delta_json = rt.compute_delta(0, None, key, flat)
            print(f"Delta size (chars): {len(delta_json)}")
            # apply delta and update master
            rt.update_master_with_delta(0, None, key, delta_json)
            print(f"Master updated for {key}")

    # Attach hooks to model
    attach_kv_hooks(model, capture_fn)

    # Simulate two forward passes with slightly different tokens
    tokens1 = torch.randint(0, 256, (1, 8))
    tokens2 = tokens1.clone()
    tokens2[0, 3] = (tokens2[0, 3] + 1) % 256  # small change to create a sparse delta

    # Run first pass (registers master)
    _ = model(tokens1)

    # Run second pass (computes delta and updates master)
    _ = model(tokens2)

if __name__ == "__main__":
    main()
