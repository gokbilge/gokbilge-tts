# Recipe: ISSAI → Piper Turkish TTS

Train a Piper-compatible VITS model on the ISSAI Turkish Speech Corpus and export to ONNX.

## Prerequisites

```bash
pip install -e "../../[train]"
pip install piper-train  # installs VITS training fork
espeak-ng --version      # must be 1.51+, with 'tr' voice
```

## Steps

### 1. Prepare dataset

```bash
bash prepare.sh /path/to/issai_corpus ./data
```

Outputs `data/issai_manifest.jsonl` (one record per utterance: `{"audio": "...", "text": "...", "phonemes": "..."}`).

### 2. Train

```bash
bash train.sh ./data ./checkpoints
```

Trains VITS using `../../configs/vits_tr_v0_1.yaml`. Logs to `checkpoints/train.log`.

### 3. Export to ONNX (Piper format)

```bash
bash export_onnx.sh ./checkpoints/epoch_1000 ./models/gokbilge-tr-medium
```

Produces `gokbilge-tr-medium.onnx` and `gokbilge-tr-medium.onnx.json`.

### 4. Run inference

```bash
bash infer.sh ./models/gokbilge-tr-medium "Bugün hava çok güzel." output.wav
```

## Evaluation

After training, run the benchmark sentences:

```bash
while IFS= read -r line; do
    bash infer.sh ./models/gokbilge-tr-medium "$line" "output_$(date +%s).wav"
done < ../../benchmarks/turkish_eval_sentences.txt
```

Evaluate perceptually using the rubric in `../../benchmarks/SCORING_RUBRIC.md`.
