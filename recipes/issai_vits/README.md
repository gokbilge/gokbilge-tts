# Recipe: ISSAI → VITS PyTorch Turkish TTS

Train a VITS model directly using the PyTorch implementation.
This recipe targets development and research use (not deployment).
For production-ready ONNX output, see `../issai_piper/`.

## Prerequisites

```bash
pip install -e "../../[train]"
# Clone the VITS reference implementation:
git clone https://github.com/jaywalnut310/vits vits_src
pip install -r vits_src/requirements.txt
```

## Steps

### 1. Prepare dataset

```bash
bash prepare.sh /path/to/issai_corpus ./data
```

### 2. Train

```bash
bash train.sh ./data ./checkpoints
```

### 3. Inference

```bash
bash infer.sh ./checkpoints/G_1000.pth "Bugün hava çok güzel." output.wav
```
