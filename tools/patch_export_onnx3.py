"""Move model to CPU before ONNX export (dummy inputs are on CPU)."""
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    src = f.read()

old = "    model = VitsModel.load_from_checkpoint(args.checkpoint, dataset=None)\n    model_g = model.model_g"
new = "    model = VitsModel.load_from_checkpoint(args.checkpoint, dataset=None, map_location=\"cpu\")\n    model_g = model.model_g.cpu()"

if old not in src:
    print("ERROR: marker not found", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched:", path)
