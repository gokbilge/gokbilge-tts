"""Add weights_only=False safe-load for torch 2.6+ in export_onnx.py."""
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    src = f.read()

# Insert safe-globals registration right after the torch import
old = "import torch\n\nfrom .vits.lightning import VitsModel"
new = (
    "import pathlib\n"
    "import torch\n"
    "\n"
    "# torch 2.6+ changed weights_only default to True; allow PosixPath in checkpoints\n"
    "torch.serialization.add_safe_globals([pathlib.PosixPath])\n"
    "\n"
    "from .vits.lightning import VitsModel"
)

if old not in src:
    print("ERROR: marker not found in export_onnx.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched:", path)
