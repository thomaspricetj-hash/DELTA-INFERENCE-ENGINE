# delta_inference/python/model_hooks.py
"""
Utilities to attach hooks to PyTorch transformer models to capture KV-cache tensors.
This is intentionally minimal and targets common LLaMA/transformer-style modules.
"""

import torch
from typing import Callable, Dict, Any, Optional

def attach_kv_hooks(model: torch.nn.Module, capture_fn: Callable[[int, Optional[int], str, torch.Tensor], None]):
    """
    Attach forward hooks to modules that produce KV tensors.
    capture_fn(layer, head, name, tensor) will be called with the tensor (float32).
    This function is conservative: it attaches to submodules named 'self_attn', 'attn', or 'kv'.
    """
    for name, module in model.named_modules():
        lname = name.lower()
        if any(k in lname for k in ("self_attn", "attn", "kv", "multiheadattention")):
            # attach a forward hook
            def make_hook(layer_name):
                def hook(module, input, output):
                    # output may be tuple; try to find tensors
                    tensors = []
                    if isinstance(output, torch.Tensor):
                        tensors = [output]
                    elif isinstance(output, (list, tuple)):
                        tensors = [o for o in output if isinstance(o, torch.Tensor)]
                    else:
                        return
                    for idx, t in enumerate(tensors):
                        # flatten to contiguous float32 numpy for delta engine
                        if not t.is_contiguous():
                            t = t.contiguous()
                        if t.dtype != torch.float32:
                            t = t.float()
                        capture_fn(layer_name, None, f"{name}_out_{idx}", t.detach())
                return hook
            module.register_forward_hook(make_hook(name))

def example_capture_fn(layer: int, head: Optional[int], name: str, tensor: torch.Tensor):
    """
    Example capture function that converts tensor to numpy and prints shape.
    Replace with DeltaRuntime.register_master / compute_delta calls.
    """
    arr = tensor.cpu().numpy()
    print(f"Captured {name} shape={arr.shape} dtype={arr.dtype}")
