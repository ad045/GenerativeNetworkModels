from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]  # project root
GNM = ROOT # / "src" / "gnm"

# A) write _device.py (idempotent)
DEVICE_HELPER = '''
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
'''.lstrip()

(GNM / "_device.py").write_text(DEVICE_HELPER)

# B) ensure exports in __init__.py
init_path = GNM / "__init__.py"
if init_path.exists():
    txt = init_path.read_text()
else:
    txt = ""
if "from ._device import get_device, to_device" not in txt:
    txt += "\nfrom ._device import get_device, to_device\n"
if "__all__" in txt:
    # Append names to __all__ if present
    txt = re.sub(r"__all__\s*=\s*\[(.*?)\]",
                 lambda m: m.group(0).rstrip("]") + ', "get_device", "to_device"]',
                 txt, flags=re.S)
else:
    txt += '\n__all__ = ["get_device", "to_device"]\n'
init_path.write_text(txt)

# C) replace common device patterns and add map_location
py_files = list(GNM.rglob("*.py"))
for p in py_files:
    s = p.read_text()

    # Replace cuda-or-cpu patterns with get_device()
    s = re.sub(r'torch\.device\(\s*"(cuda:0|cuda)"\s*if\s*torch\.cuda\.is_available\(\)\s*else\s*"(cpu)"\s*\)',
               'get_device()', s)
    s = re.sub(r'torch\.device\(\s*"(cuda)"\s*if\s*torch\.cuda\.is_available\(\)\s*else\s*"(cpu)"\s*\)',
               'get_device()', s)

    # Ensure we import get_device where torch.device was used
    if "get_device()" in s and "from ._device import get_device" not in s:
        # add a safe import near the top
        s = re.sub(r"(^from .+|^import .+)", r"\g<0>", s, count=0, flags=re.M)  # no-op, just anchor
        s = "from ._device import get_device\n" + s

    # Add map_location to torch.load calls that lack it
    s = re.sub(r'torch\.load\(\s*([^)]+?)\s*\)',
               r'torch.load(\1, map_location=get_device(, map_location=get_device()))', s)

    # Add device kwarg to simple function defs if missing (lightweight)
    s = re.sub(r"def ([a-zA-Z_][\w]*)\(([^)]*)\):",
               lambda m: (
                   f'def {m.group(1)}({m.group(2)}{", device=None" if "device=" not in m.group(2) else ""}):'
               ),
               s)

    p.write_text(s)

print("Applied macOS/MPS patches to GNM.")

# /Users/adrian/Documents/01_projects/14_4D_lab/src/imported_libraries/GenerativeNetworkModels_2/src/gnm/_device.py