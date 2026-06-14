"""Turkish abbreviation expansion.

Expands common Turkish abbreviations to their spoken forms.

Examples:
    Dr.    -> "doktor"
    Prof.  -> "profesör"
    Sn.    -> "sayın"
    No.    -> "numara"
    TL     -> "Türk lirası"
    km     -> "kilometre"
"""

from __future__ import annotations

import re

# Sorted at module level — longest first to prevent shorter entries shadowing longer ones.
# Suffix handling (TL'ye -> "Türk lirasına") is TODO: vowel harmony on expansion is needed.
ABBREVIATIONS: dict[str, str] = {
    # Academic / professional titles
    "Dr.": "doktor",
    "Prof.": "profesör",
    "Doç.": "doçent",
    "Sn.": "sayın",
    # Common abbreviations
    "No.": "numara",
    "vs.": "vesaire",
    "vb.": "ve benzeri",
    "vd.": "ve diğerleri",
    "bkz.": "bakınız",
    "Bkz.": "bakınız",
    # Currencies — expanded BEFORE number normalization runs, so "50 TL" -> "50 Türk lirası" -> "elli Türk lirası"
    "TL": "Türk lirası",
    "USD": "dolar",
    "EUR": "avro",
    # Units — longer variants first
    "kHz": "kilohertz",
    "MHz": "megahertz",
    "GHz": "gigahertz",
    "km": "kilometre",
    "cm": "santimetre",
    "mm": "milimetre",
    "kg": "kilogram",
    "Hz": "hertz",
    "MB": "megabayt",
    "GB": "gigabayt",
    "TB": "terabayt",
    "ml": "mililitre",
    "lt": "litre",
    "m": "metre",
    "g": "gram",
}

# Pre-sorted once: longest abbreviation first.
_SORTED_ABBREVS = sorted(ABBREVIATIONS.items(), key=lambda kv: len(kv[0]), reverse=True)


def normalize_abbreviations(text: str) -> str:
    """Expand abbreviations in Turkish text using word-boundary-aware replacement."""
    for abbrev, expansion in _SORTED_ABBREVS:
        escaped = re.escape(abbrev)
        if abbrev[-1].isalpha():
            # Non-period ending (TL, km, etc.): require full word boundary on both sides.
            pattern = r'\b' + escaped + r'\b'
        else:
            # Period-ending (Dr., Prof., etc.): word boundary only at start;
            # the trailing period is part of the abbreviation.
            pattern = r'\b' + escaped
        text = re.sub(pattern, expansion, text)
    return text
