"""Turkish abbreviation expansion.

Expands common Turkish abbreviations to their spoken forms.

Examples:
    Dr.    -> "doktor"
    Prof.  -> "profesör"
    TL     -> "lira"
    km     -> "kilometre"
    m²     -> "metrekare"
"""

from __future__ import annotations

# TODO: build comprehensive Turkish abbreviation lexicon
ABBREVIATIONS: dict[str, str] = {
    "Dr.": "doktor",
    "Prof.": "profesör",
    "Doç.": "doçent",
    "Arş. Gör.": "araştırma görevlisi",
    "Öğr. Gör.": "öğretim görevlisi",
    "TL": "lira",
    "USD": "dolar",
    "EUR": "avro",
    "km": "kilometre",
    "m": "metre",
    "cm": "santimetre",
    "mm": "milimetre",
    "kg": "kilogram",
    "g": "gram",
    "ml": "mililitre",
    "lt": "litre",
    # TODO: add units, institutions, common Turkish abbreviations
}


def normalize_abbreviations(text: str) -> str:
    """Expand abbreviations in Turkish text.

    TODO:
        - Context-sensitive expansion (m. = metre or madde?)
        - Suffix handling: km'ye -> "kilometreye"
        - Uppercase/lowercase variants
        - Institution names: TBMM, TRT, TÜBİTAK
        - Acronyms read letter by letter vs. as words
    """
    # Placeholder — returns text unchanged
    return text
