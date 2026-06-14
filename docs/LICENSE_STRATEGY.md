# License Strategy

## Code — MIT

The gokbilge-tts codebase (`src/`) is released under MIT. This permits:
- Commercial use
- Modification
- Distribution
- Private use

## Models — depends on training data license

### ISSAI Turkish Speech Corpus

Before releasing any model weights derived from ISSAI data:
1. Read the full ISSAI license at the HuggingFace dataset page
2. Confirm whether commercial use of derivative models is permitted
3. Confirm attribution requirements

If ISSAI allows it, release model weights under Apache 2.0 or MIT.
If ISSAI restricts commercial use, release model weights under CC BY-NC 4.0.

**Never assume — read the license before uploading model weights.**

## Third-party dependencies

| Package | License | Notes |
|---------|---------|-------|
| PyTorch | BSD-style | OK for commercial |
| VITS (jaywalnut310) | MIT | OK |
| Piper-train | MIT | OK |
| espeak-ng | GPL-3.0 | Backend only — if linking, check GPL implications |
| phonemizer | GPL-3.0 | Same as espeak-ng |
| onnxruntime | MIT | OK for deployment |

**Note on espeak-ng / phonemizer:** These are GPL-3.0. Using them as an external process (subprocess) is generally fine under GPL. Linking them statically into a commercial product may require GPL compliance. The fallback G2P (`g2p/turkish.py`) is MIT and has no such concern — prefer it for deployment.

## Attribution

When using or fine-tuning this model, please cite:
- ISSAI dataset authors
- VITS paper (Kim et al., 2021)
- This repository (gokbilge/gokbilge-tts)
