"""Turkish date normalization.

Converts date expressions to Turkish spoken form.

Examples:
    14.06.2026       -> "on dört haziran iki bin yirmi altı"
    Ocak 2026        -> "ocak iki bin yirmi altı"
    2026 yılı        -> "iki bin yirmi altı yılı"
"""

from __future__ import annotations

MONTHS_TR = {
    1: "ocak", 2: "şubat", 3: "mart", 4: "nisan",
    5: "mayıs", 6: "haziran", 7: "temmuz", 8: "ağustos",
    9: "eylül", 10: "ekim", 11: "kasım", 12: "aralık",
}


def normalize_dates(text: str) -> str:
    """Replace date expressions with Turkish word equivalents.

    TODO:
        - DD.MM.YYYY format
        - DD/MM/YYYY format
        - Month name + year
        - Partial dates (just month, just year)
        - Date ranges
        - Turkish month name casing (suffix-based: Ocak'ta, Haziran'da)
    """
    # Placeholder — returns text unchanged
    return text
