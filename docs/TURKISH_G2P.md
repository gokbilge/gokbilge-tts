# Turkish G2P (Grapheme-to-Phoneme)

## Why Turkish G2P is relatively easy

Turkish has near-perfect phonemic orthography: one grapheme maps to exactly one phoneme, with very few exceptions. This makes rule-based G2P highly accurate for native Turkish words.

Contrast with English: "read" /riːd/ vs "read" /rɛd/ — context-dependent. Turkish has essentially no homographs.

## Phoneme inventory

### Vowels (8)

| Letter | IPA | Example |
|--------|-----|---------|
| a | /a/ | araba |
| e | /e/ | ev |
| i | /i/ | ip |
| ı | /ɯ/ | ışık (back unrounded — no English equivalent) |
| o | /o/ | okul |
| ö | /ø/ | göz (front rounded — like German ö) |
| u | /u/ | ulus |
| ü | /y/ | üzüm (front rounded — like German ü) |

### Consonants (21)

| Letter | IPA | Notes |
|--------|-----|-------|
| b | /b/ | |
| c | /dʒ/ | NOT /k/ — like English "j" in "jar" |
| ç | /tʃ/ | like English "ch" in "chair" |
| d | /d/ | |
| f | /f/ | |
| g | /ɡ/ | |
| ğ | — | soft g: silent or lengthens preceding vowel |
| h | /h/ | |
| j | /ʒ/ | like English "s" in "measure" |
| k | /k/ | |
| l | /l/ | |
| m | /m/ | |
| n | /n/ | |
| p | /p/ | |
| r | /ɾ/ | flap, not English rhotic |
| s | /s/ | |
| ş | /ʃ/ | like English "sh" in "ship" |
| t | /t/ | |
| v | /v/ | |
| y | /j/ | like English "y" in "yes" |
| z | /z/ | |

## The soft-g (ğ) rule

`ğ` is the only letter with non-trivial behavior:

- Between vowels: **silent** — "dağ" → /da/ (with lengthened /a/)
- Before consonant or end of word: **lengthens preceding vowel**
- Never appears at the start of a word

Current implementation in `g2p/turkish.py`: silences ğ in all positions (approximation). Vowel lengthening is a TODO.

## Vowel harmony

Turkish vowels follow strict front/back and round/unround harmony in suffixes. The G2P does not need to handle this for the lexical form — harmony only applies to suffixes, and the base word's vowels are always spelled out explicitly.

## Allophonic variation (not currently modeled)

- /k/ → [c] before front vowels (palatalization)
- /l/ → dark [ɫ] before back vowels
- Final devoicing: /b d ɡ/ → [p t k] in certain positions

These are fine phonetic details. The VITS model may learn them implicitly from audio.

## espeak-ng fallback

For borrowed words with irregular phoneme patterns (İngilizce loanwords, proper nouns), the rule-based table fails gracefully. When `phonemizer` is installed, the espeak-ng backend can be used via `g2p/phonemizer.py` for cross-checking.
