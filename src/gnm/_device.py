# from ._device import get_device

from __future__ import annotations
import torch, os

def prefer_mps() -> bool:
    return os.environ.get("GNM_DISABLE_MPS", "0") not in {"1","true","True"}

def get_device() -> torch.device:
    if prefer_mps() and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_device(x, device=None):
    if device is None: device = get_device()
    try:
        return x.to(device)
    except AttributeError:
        return x
