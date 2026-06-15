"""Pure-Python stub for piper_phonemize.

Replaces the unavailable C extension (no aarch64 wheel, OnnxRuntime C++ headers
required to build from source). Uses espeak-ng subprocess for phonemization and
the phoneme→ID map extracted verbatim from piper-phonemize/src/phoneme_ids.hpp.

Install: copy to site-packages as piper_phonemize.py  (or as a package __init__.py)
Requires: espeak-ng binary on PATH.
"""
from __future__ import annotations

import subprocess
from collections import Counter
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# DEFAULT_PHONEME_ID_MAP extracted from piper-phonemize/src/phoneme_ids.hpp
# (DEFAULT_PHONEME_ID_MAP / MAX_PHONEMES = 256)
# ---------------------------------------------------------------------------

_ESPEAK_MAP: Dict[str, List[int]] = {
    '_': [0], '^': [1], '$': [2], ' ': [3], '!': [4], "'": [5],
    '(': [6], ')': [7], ',': [8], '-': [9], '.': [10], ':': [11],
    ';': [12], '?': [13],
    'a': [14], 'b': [15], 'c': [16], 'd': [17], 'e': [18], 'f': [19],
    'h': [20], 'i': [21], 'j': [22], 'k': [23], 'l': [24], 'm': [25],
    'n': [26], 'o': [27], 'p': [28], 'q': [29], 'r': [30], 's': [31],
    't': [32], 'u': [33], 'v': [34], 'w': [35], 'x': [36], 'y': [37],
    'z': [38],
    'æ': [39], 'ç': [40], 'ð': [41], 'ø': [42], 'ħ': [43], 'ŋ': [44],
    'œ': [45], 'ǀ': [46], 'ǁ': [47], 'ǂ': [48], 'ǃ': [49],
    'ɐ': [50], 'ɑ': [51], 'ɒ': [52], 'ɓ': [53], 'ɔ': [54], 'ɕ': [55],
    'ɖ': [56], 'ɗ': [57], 'ɘ': [58], 'ə': [59], 'ɚ': [60], 'ɛ': [61],
    'ɜ': [62], 'ɞ': [63], 'ɟ': [64], 'ɠ': [65], 'ɡ': [66], 'ɢ': [67],
    'ɣ': [68], 'ɤ': [69], 'ɥ': [70], 'ɦ': [71], 'ɧ': [72], 'ɨ': [73],
    'ɪ': [74], 'ɫ': [75], 'ɬ': [76], 'ɭ': [77], 'ɮ': [78], 'ɯ': [79],
    'ɰ': [80], 'ɱ': [81], 'ɲ': [82], 'ɳ': [83], 'ɴ': [84], 'ɵ': [85],
    'ɶ': [86], 'ɸ': [87], 'ɹ': [88], 'ɺ': [89], 'ɻ': [90], 'ɽ': [91],
    'ɾ': [92], 'ʀ': [93], 'ʁ': [94], 'ʂ': [95], 'ʃ': [96], 'ʄ': [97],
    'ʈ': [98], 'ʉ': [99], 'ʊ': [100], 'ʋ': [101], 'ʌ': [102], 'ʍ': [103],
    'ʎ': [104], 'ʏ': [105], 'ʐ': [106], 'ʑ': [107], 'ʒ': [108], 'ʔ': [109],
    'ʕ': [110], 'ʘ': [111], 'ʙ': [112], 'ʛ': [113], 'ʜ': [114], 'ʝ': [115],
    'ʟ': [116], 'ʡ': [117], 'ʢ': [118], 'ʲ': [119], 'ˈ': [120], 'ˌ': [121],
    'ː': [122], 'ˑ': [123], '˞': [124], 'β': [125], 'θ': [126], 'χ': [127],
    'ᵻ': [128], 'ⱱ': [129],
    '0': [130], '1': [131], '2': [132], '3': [133], '4': [134], '5': [135],
    '6': [136], '7': [137], '8': [138], '9': [139],
    '̧': [140],  # combining cedilla
    '̃': [141],  # combining tilde
    '̪': [142],  # combining bridge below
    '̯': [143],  # combining inverted breve below
    '̩': [144],  # combining vertical line below
    'ʰ': [145], 'ˤ': [146], 'ε': [147], '↓': [148], '#': [149], '"': [150],
    '↑': [151],
    '̺': [152],  # combining retroflex hook below
    '̻': [153],  # combining square below
    'g': [154], 'ʦ': [155], 'X': [156],
    '̝': [157],  # combining up tack below
    '̊': [158],  # combining ring above
}

_MAX_PHONEMES = 256
_BOS_ID = 1   # '^'
_EOS_ID = 2   # '$'
_PAD_ID = 0   # '_'


# ---------------------------------------------------------------------------
# Public API (matches piper_phonemize C extension interface)
# ---------------------------------------------------------------------------

def phonemize_espeak(
    text: str,
    voice: str,
    clause_breaker: Optional[str] = None,
) -> List[List[str]]:
    """Phonemize text with espeak-ng, returning per-sentence phoneme lists.

    Each phoneme is a single Unicode codepoint (matching piper convention).
    """
    try:
        proc = subprocess.run(
            ["espeak-ng", "-v", voice, "--ipa", "-q", "--"],
            input=text,
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
        )
        output = proc.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return [[]]

    sentences: List[List[str]] = []
    for line in output.splitlines():
        line = line.strip()
        if line:
            sentences.append(list(line))   # each Unicode char is one phoneme

    return sentences or [[]]


def phoneme_ids_espeak(
    phonemes: List[str],
    missing_phonemes: Optional[Counter] = None,
) -> List[int]:
    """Convert flat phoneme list → piper ID sequence (BOS/pad-interspersed/EOS).

    Matches PhonemeIdConfig defaults: addBos=true, addEos=true, interspersePad=true.
    """
    if missing_phonemes is None:
        missing_phonemes = Counter()

    ids: List[int] = [_BOS_ID, _PAD_ID]
    for ph in phonemes:
        ph_ids = _ESPEAK_MAP.get(ph)
        if ph_ids is None:
            missing_phonemes[ph] += 1
            continue
        for pid in ph_ids:
            ids.append(pid)
            ids.append(_PAD_ID)
    ids.append(_EOS_ID)
    return ids


def get_espeak_map() -> Dict[str, List[int]]:
    """Return the global espeak phoneme→ID map (language-independent)."""
    return dict(_ESPEAK_MAP)


def get_max_phonemes() -> int:
    return _MAX_PHONEMES


# ---------------------------------------------------------------------------
# Stubs for TEXT-mode phonemization (not used for Turkish/espeak pipeline)
# ---------------------------------------------------------------------------

def phonemize_codepoints(text: str) -> List[List[str]]:
    return [list(text)]


def phoneme_ids_codepoints(
    language: str,
    phonemes: List[str],
    missing_phonemes: Optional[Counter] = None,
) -> List[int]:
    return []


def get_codepoints_map() -> Dict[str, Dict[str, List[int]]]:
    return {}


def tashkeel_run(text: str) -> str:
    """No-op stub — Arabic diacritization, not needed for Turkish."""
    return text
