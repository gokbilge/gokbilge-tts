# Perceptual Scoring Rubric

Use this rubric to evaluate synthesized audio for the five benchmark sentences.
Score each dimension 1–5. Report mean overall score per model checkpoint.

## Dimensions

### 1. Intelligibility (can you understand the words?)

| Score | Description |
|-------|-------------|
| 5 | Perfect — every word clear |
| 4 | Minor errors or mumbling on one word |
| 3 | Roughly correct but 2–3 words unclear |
| 2 | Heavily distorted, hard to follow |
| 1 | Unintelligible |

### 2. Phoneme Accuracy (are Turkish-specific sounds correct?)

Focus on: ç /tʃ/, ş /ʃ/, ğ (silent/lengthening), ı /ɯ/, ö /ø/, ü /y/

| Score | Description |
|-------|-------------|
| 5 | All Turkish phonemes correct |
| 4 | One phoneme slightly off (e.g. ı sounds like i) |
| 3 | Two phonemes wrong — noticeable foreign accent |
| 2 | Most Turkish-specific sounds wrong |
| 1 | No Turkish phonemes recognizable |

### 3. Prosody (rhythm and naturalness)

| Score | Description |
|-------|-------------|
| 5 | Natural Turkish sentence rhythm |
| 4 | Mostly natural, minor flat/choppy sections |
| 3 | Noticeably robotic but acceptable |
| 2 | Choppy or unnatural at multiple points |
| 1 | Monotone or highly fragmented |

### 4. Audio Quality (noise, artifacts)

| Score | Description |
|-------|-------------|
| 5 | Clean — no artifacts |
| 4 | Very slight noise, not distracting |
| 3 | Audible noise or occasional glitch |
| 2 | Significant noise throughout |
| 1 | Severe distortion or clicks |

## Scoring Sheet Template

```
Model: _______________  Checkpoint: _______________  Date: ___________

Sentence 1: Bugün hava çok güzel.
  Intelligibility: /5   Phoneme: /5   Prosody: /5   Quality: /5   Mean: /5

Sentence 2: Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu.
  Intelligibility: /5   Phoneme: /5   Prosody: /5   Quality: /5   Mean: /5

Sentence 3: Çocuklar çiçek, şeker ve üzüm yedi.
  Intelligibility: /5   Phoneme: /5   Prosody: /5   Quality: /5   Mean: /5

Sentence 4: Öğrenciler ölçüm sonuçlarını değerlendirdi.
  Intelligibility: /5   Phoneme: /5   Prosody: /5   Quality: /5   Mean: /5

Sentence 5: Şirket yüzde otuz beş büyüme açıkladı.
  Intelligibility: /5   Phoneme: /5   Prosody: /5   Quality: /5   Mean: /5

OVERALL MEAN: /5
Notes:
```
