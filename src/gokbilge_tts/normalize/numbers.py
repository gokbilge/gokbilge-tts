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
SCALE = ["", "bin", "milyon", "milyar", "trilyon"]


def _int_to_words(n: int) -> str:
    """Convert a non-negative integer to Turkish words."""
    # TODO: implement full conversion including edge cases (sıfır, negative, large numbers)
    if n == 0:
        return "sıfır"
    raise NotImplementedError("Full number-to-words not yet implemented")


def normalize_numbers(text: str) -> str:
    """Replace numeric expressions in text with Turkish word equivalents.

    TODO:
        - Cardinal numbers: 42 -> "kırk iki"
        - Ordinal numbers: 3. -> "üçüncü"
        - Percentages: %35 -> "yüzde otuz beş"
        - Decimals: 3,14 -> "üç virgül on dört"
        - Large numbers: 1.000.000 -> "bir milyon"
        - Years: 1923 -> "bin dokuz yüz yirmi üç"
        - Phone numbers: handled separately
        - Currency: 50 TL -> "elli lira"
    """
    # Placeholder — returns text unchanged
    return text
