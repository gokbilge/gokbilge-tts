"""Turkish grapheme-to-phoneme (G2P) conversion.

Rule-based first version. Handles straightforward cases directly from orthography.
Turkish orthography is largely phonemic — one letter, one sound — making rule-based
G2P achievable for most words without a pronunciation dictionary.

Known challenges:
    - ğ (soft g): context-dependent lengthening or /j/
    - k: palatalized /c/ before front vowels
    - l: clear vs. dark allophone
    - Loanword phonology (foreign words may not follow Turkish rules)
    - Proper nouns and abbreviations
"""

from __future__ import annotations

# Straightforward 1:1 grapheme -> phoneme mappings
GRAPHEME_TO_PHONEME: dict[str, str] = {
    # Vowels
    "a": "a",
    "e": "e",
    "i": "i",
    "ı": "ɯ",    # back unrounded — the most problematic for non-native TTS
    "o": "o",
    "ö": "ø",
    "u": "u",
    "ü": "y",
    # Consonants — straightforward
    "b": "b",
    "c": "dʒ",   # ALWAYS /dʒ/ — never /k/
    "ç": "tʃ",
    "d": "d",
    "f": "f",
    "g": "g",
    "ğ": "",   # context-dependent (handled by _handle_soft_g); entry satisfies completeness check
    "h": "h",
    "j": "ʒ",
    "k": "k",    # TODO: palatalize to /c/ before front vowels
    "l": "l",    # TODO: dark /ɫ/ before back vowels
    "m": "m",
    "n": "n",
    "p": "p",
    "r": "ɾ",    # tap r
    "s": "s",
    "ş": "ʃ",
    "t": "t",
    "v": "v",
    "y": "j",
    "z": "z",
}

FRONT_VOWELS = set("eiöü")
BACK_VOWELS = set("aıou")


def _handle_soft_g(prev_char: str, next_char: str) -> str:
    """Convert ğ (soft g) based on surrounding vowels.

    Rules:
        - After back vowel: lengthens the preceding vowel (silent)
        - After front vowel: /j/ or lengthens
        - Between two vowels: usually lengthens first vowel
    TODO: refine with more data — current rule is approximate.
    """
    if prev_char in FRONT_VOWELS:
        return "j"   # dağ -> da: but değer -> dejer
    return ""        # Silent / vowel lengthening (represented by dropping)


def text_to_phonemes(text: str) -> str:
    """Convert Turkish text to a phoneme string.

    Args:
        text: Normalized Turkish text (after normalize_text()).

    Returns:
        Space-separated phoneme sequence.

    TODO:
        - Syllabification
        - Stress marking
        - Vowel harmony validation
        - Loanword handling
        - Proper noun handling
        - Suffix boundary apostrophes (Ankara'da)
        - Final devoicing
    """
    text = text.lower()
    phonemes: list[str] = []
    chars = list(text)

    i = 0
    while i < len(chars):
        ch = chars[i]
        prev_ch = chars[i - 1] if i > 0 else ""
        next_ch = chars[i + 1] if i < len(chars) - 1 else ""

        if ch == "ğ":
            p = _handle_soft_g(prev_ch, next_ch)
        elif ch in GRAPHEME_TO_PHONEME:
            p = GRAPHEME_TO_PHONEME[ch]
        elif ch == " ":
            p = "_"
        elif ch in ".,!?;:-":
            p = "~"
        else:
            p = ch  # pass through unknowns

        if p:
            phonemes.append(p)
        i += 1

    return " ".join(phonemes)
