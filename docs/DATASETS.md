# Datasets

## ISSAI Turkish Speech Corpus

**Primary training dataset for gokbilge-tts v0.1.**

- **Source**: `issai/Turkish_Speech_Corpus` on HuggingFace (or direct download)
- **Size**: ~10 hours of read speech
- **Speaker(s)**: Professional Turkish speaker(s)
- **Audio format**: 16-bit PCM WAV, 16 kHz or 22 kHz (varies by subset)
- **Transcripts**: Clean orthographic Turkish text
- **License**: Check ISSAI license terms before commercial use

### Preprocessing

1. Resample to 22050 Hz (if needed)
2. Trim leading/trailing silence
3. Text normalization (see `docs/TEXT_NORMALIZATION.md`)
4. G2P phonemization (see `docs/TURKISH_G2P.md`)
5. Filter outliers: duration < 0.5 s or > 20 s, text length > 190 chars

### Manifest format

Each line is a JSON record:

```json
{"audio": "data/wav/sentence_001.wav", "text": "Bugün hava çok güzel.", "phonemes": "b u ɡ y n h a v a tʃ o k ɡ y z e l", "duration": 1.84, "speaker_id": 0}
```

## Other Turkish Speech Corpora (not yet integrated)

| Name | Hours | License | Notes |
|------|-------|---------|-------|
| Mozilla Common Voice (tr) | ~100h | CC0 | Crowdsourced, noisy |
| TTC (METU) | ~20h | Research | Requires institutional agreement |
| VoxLingua107 (tr slice) | ~12h | CC-BY | Wild speech, not read |

Contributions to integrate additional corpora are welcome — see `CONTRIBUTING.md` (TBD).
