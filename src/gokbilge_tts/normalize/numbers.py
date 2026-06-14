"""Turkish number normalization.

Converts digits and numeric expressions to Turkish word form.

Examples:
    1923       -> "bin dokuz yüz yirmi üç"
    2.5        -> "iki virgül beş"
    %35        -> "yüzde otuz beş"
    1.000.000  -> "bir milyon"
"""

from __future__ import annotations

import re

ONES = [
    "", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz",
]
TENS = [
    "", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan",
]

_VOWELS_BACK_UNROUNDED = frozenset("aı")
_VOWELS_FRONT_UNROUNDED = frozenset("ei")
_VOWELS_BACK_ROUNDED = frozenset("ou")
_VOWELS_FRONT_ROUNDED = frozenset("öü")


def _int_to_words(n: int) -> str:
    """Convert a non-negative integer to Turkish words."""
    if n == 0:
        return "sıfır"
    if n < 0:
        return "eksi " + _int_to_words(-n)

    parts: list[str] = []
    rem = n

    if rem >= 1_000_000_000_000:
        q, rem = divmod(rem, 1_000_000_000_000)
        parts.append(_int_to_words(q) + " trilyon")

    if rem >= 1_000_000_000:
        q, rem = divmod(rem, 1_000_000_000)
        parts.append(_int_to_words(q) + " milyar")

    if rem >= 1_000_000:
        q, rem = divmod(rem, 1_000_000)
        parts.append(_int_to_words(q) + " milyon")

    if rem >= 1_000:
        q, rem = divmod(rem, 1_000)
        # "bin" not "bir bin" for 1000
        if q == 1:
            parts.append("bin")
        else:
            parts.append(_int_to_words(q) + " bin")

    if rem >= 100:
        q, rem = divmod(rem, 100)
        # "yüz" not "bir yüz" for 100
        if q == 1:
            parts.append("yüz")
        else:
            parts.append(ONES[q] + " yüz")

    if rem >= 10:
        q, rem = divmod(rem, 10)
        parts.append(TENS[q])

    if rem > 0:
        parts.append(ONES[rem])

    return " ".join(parts)


def _ordinal_suffix(word: str) -> str:
    """Return the vowel-harmony-correct ordinal suffix for the given Turkish word."""
    for ch in reversed(word.replace(" ", "")):
        if ch in _VOWELS_BACK_UNROUNDED:
            return "ıncı"
        if ch in _VOWELS_FRONT_UNROUNDED:
            return "inci"
        if ch in _VOWELS_BACK_ROUNDED:
            return "uncu"
        if ch in _VOWELS_FRONT_ROUNDED:
            return "üncü"
    return "inci"


def _int_to_ordinal(n: int) -> str:
    """Convert integer to Turkish ordinal (e.g. 3 -> 'üçüncü')."""
    word = _int_to_words(n)
    return word + _ordinal_suffix(word)


# Precompiled patterns — ordered to avoid conflicts.
# Apply in sequence: more specific patterns before general ones.
_RE_PERCENT = re.compile(r'%\s*(\d+)')
_RE_THOUSANDS_DECIMAL = re.compile(r'\b(\d{1,3}(?:\.\d{3})+),(\d+)\b')
_RE_THOUSANDS = re.compile(r'\b(\d{1,3}(?:\.\d{3})+)\b')
_RE_DECIMAL_COMMA = re.compile(r'\b(\d+),(\d+)\b')
_RE_ORDINAL = re.compile(r'\b(\d+)\.(?=[ \t])')
_RE_INTEGER = re.compile(r'\b(\d+)\b')


def normalize_numbers(text: str) -> str:
    """Replace numeric expressions in text with Turkish word equivalents."""
    # %35 or % 35 -> "yüzde otuz beş"
    text = _RE_PERCENT.sub(
        lambda m: "yüzde " + _int_to_words(int(m.group(1))),
        text,
    )

    # 1.000,50 -> "bin virgül elli" (thousands separator + decimal)
    text = _RE_THOUSANDS_DECIMAL.sub(
        lambda m: (
            _int_to_words(int(m.group(1).replace(".", "")))
            + " virgül "
            + _int_to_words(int(m.group(2)))
        ),
        text,
    )

    # 1.000.000 -> "bir milyon" (Turkish thousands separator)
    text = _RE_THOUSANDS.sub(
        lambda m: _int_to_words(int(m.group(0).replace(".", ""))),
        text,
    )

    # 3,14 -> "üç virgül on dört" (Turkish decimal comma)
    text = _RE_DECIMAL_COMMA.sub(
        lambda m: _int_to_words(int(m.group(1))) + " virgül " + _int_to_words(int(m.group(2))),
        text,
    )

    # "3. " -> "üçüncü " (ordinal, only before whitespace)
    text = _RE_ORDINAL.sub(
        lambda m: _int_to_ordinal(int(m.group(1))),
        text,
    )

    # 1923 -> "bin dokuz yüz yirmi üç" (remaining plain integers)
    text = _RE_INTEGER.sub(
        lambda m: _int_to_words(int(m.group(1))),
        text,
    )

    return text
