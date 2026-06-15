"""Add dynamo=False to torch.onnx.export call to force legacy TorchScript exporter."""
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    src = f.read()

# Add dynamo=False after verbose=False line
old = "        verbose=False,\n        opset_version=OPSET_VERSION,"
new = "        verbose=False,\n        dynamo=False,  # legacy TorchScript exporter; dynamo exporter fails on mixed device VITS models\n        opset_version=OPSET_VERSION,"

if old not in src:
    print("ERROR: marker not found", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched:", path)
