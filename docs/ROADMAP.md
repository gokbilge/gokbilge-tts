# Roadmap

## v0.1 — First working model

- [ ] ISSAI dataset preparation pipeline (`prepare_issai.py`)
- [ ] Text normalization: numbers, dates, abbreviations
- [ ] Rule-based Turkish G2P
- [ ] VITS training recipe (ISSAI Piper)
- [ ] Piper ONNX export
- [ ] Benchmark scores on 5-sentence eval set
- [ ] HuggingFace model release (gokbilge-tts/gokbilge-tr-medium-v0.1)

## v0.2 — Quality improvements

- [ ] espeak-ng G2P backend for borrowed words
- [ ] Number normalization fully implemented (not placeholder)
- [ ] Date normalization fully implemented
- [ ] VITS2 architecture evaluation
- [ ] MOS prediction (UTMOS) on eval set
- [ ] ASR-based Character Error Rate evaluation (Whisper)

## v0.3 — Multi-speaker

- [ ] Evaluate speaker count in ISSAI (multiple speakers available)
- [ ] Speaker embedding (d-vector or ECAPA-TDNN)
- [ ] Multi-speaker VITS training
- [ ] Speaker-conditioned inference

## Future

- Streaming inference
- SSML subset support (rate, pitch, volume)
- Low-resource dialect adaptation
- Emotional/expressive TTS
