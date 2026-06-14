# Text Normalization

Turkish TTS requires careful text normalization before G2P. The model has no digit handling — all numbers and abbreviations must be spelled out in full before phonemization.

## Pipeline

```
Raw text
  → NFC unicode normalization
  → Remove non-speech characters (control chars, ZWS, etc.)
  → Normalize punctuation (full-width → ASCII, collapse whitespace)
  → Expand abbreviations
  → Expand dates
  → Expand numbers
  → Normalized text ready for G2P
```

## Numbers

Turkish cardinal numbers are agglutinative and require morphological agreement in compound forms. For TTS, we expand to canonical written form (no suffix inflection):

| Input | Output |
|-------|--------|
| 1 | bir |
| 15 | on beş |
| 35 | otuz beş |
| 100 | yüz |
| 1923 | bin dokuz yüz yirmi üç |
| 1.000.000 | bir milyon |

**Note:** Always write numbers as words in TTS input. Digits that slip through produce incorrect phoneme sequences because the G2P has no digit-to-phoneme rule.

## Dates

| Input | Output |
|-------|--------|
| 29 Ekim | yirmi dokuz Ekim |
| 15/03/2024 | on beş Mart iki bin yirmi dört |

Month names are already Turkish words and pass through G2P unchanged.

## Abbreviations

Common Turkish abbreviations:

| Abbreviation | Expansion |
|-------------|-----------|
| Dr. | Doktor |
| Prof. | Profesör |
| Müh. | Mühendis |
| TL | Türk lirası |
| km | kilometre |
| cm | santimetre |
| m² | metrekare |
| vb. | ve benzeri |
| vd. | ve diğerleri |

## Known Limitations

- Ordinal numbers not yet implemented (1., 2. → birinci, ikinci)
- Roman numerals not handled
- Foreign-language text (English, etc.) is not detected — will be phonemized as Turkish
- Currency symbols ($, €) not expanded
