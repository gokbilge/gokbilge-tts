"""Turkish phoneme symbol set for Gokbilge TTS.

Based on Turkish phonology. Each symbol maps to a distinct phoneme in Standard Turkish.
"""

# Vowels
VOWELS = [
    "a",   # /a/  — kara, bal
    "e",   # /e/  — et, ekmek
    "i",   # /i/  — ip, bir
    "ı",   # /ɯ/  — ırmak, kız   (back unrounded — no equivalent in English/German)
    "o",   # /o/  — on, bor
    "ö",   # /ø/  — öz, göz       (front rounded)
    "u",   # /u/  — un, buz
    "ü",   # /y/  — üç, gün       (front rounded)
]

# Consonants
CONSONANTS = [
    "b",   # /b/  — bal
    "c",   # /dʒ/ — cam, ceviz    (IMPORTANT: always /dʒ/, never /k/)
    "ç",   # /tʃ/ — çay, çiçek
    "d",   # /d/  — dal
    "f",   # /f/  — fal
    "g",   # /g/  — gün, gemi
    "ğ",   # /ɣ/ or silent — dağ  (soft g: lengthens preceding vowel or is silent)
    "h",   # /h/  — hava
    "j",   # /ʒ/  — jale, jilet   (French j)
    "k",   # /k/  or /c/ — kız, köy
    "l",   # /l/  or /ɫ/ — lale, bal
    "m",   # /m/  — mal
    "n",   # /n/  — ne
    "p",   # /p/  — pala
    "r",   # /ɾ/  — rüya (tap r, not English rhotic)
    "s",   # /s/  — sel
    "ş",   # /ʃ/  — şeker, şişe
    "t",   # /t/  — taş
    "v",   # /v/  — val
    "y",   # /j/  — yıl, yüz
    "z",   # /z/  — zil
]

# Special symbols
SPECIAL = [
    "_",   # silence / pause
    "~",   # sentence boundary
]

ALL_SYMBOLS = VOWELS + CONSONANTS + SPECIAL

# Turkish-specific notes for G2P implementation
# TODO: handle the following allophonic variations
# - k before front vowels: /c/ (palatalized)
# - l before front vowels: clear /l/ vs dark /ɫ/ elsewhere
# - ğ after back vowels: lengthens vowel (dağ /daː/)
# - ğ after front vowels: /j/ or lengthens vowel (değer /dejer/ or /deːer/)
# - final devoicing: d->t, b->p, g->k, c->ç in some positions
