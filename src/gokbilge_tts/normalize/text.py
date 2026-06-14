"""Top-level Turkish text normalization pipeline."""

from __future__ import annotations

from gokbilge_tts.normalize.punctuation import normalize_punctuation, remove_non_speech
from gokbilge_tts.normalize.abbreviations import normalize_abbreviations
from gokbilge_tts.normalize.dates import normalize_dates
from gokbilge_tts.normalize.numbers import normalize_numbers


def normalize_text(text: str) -> str:
    """Full Turkish text normalization pipeline.

    Applies in order:
        1. Remove non-speech elements (URLs, HTML)
        2. Punctuation / Unicode normalization
        3. Abbreviation expansion
        4. Date normalization
        5. Number normalization

    Args:
        text: Raw Turkish input text.

    Returns:
        Normalized text suitable for G2P / TTS synthesis.

    Example:
        >>> normalize_text("Şirket %35 büyüdü.")
        "Şirket yüzde otuz beş büyüdü."
    """
    text = remove_non_speech(text)
    text = normalize_punctuation(text)
    text = normalize_abbreviations(text)
    text = normalize_dates(text)
    text = normalize_numbers(text)
    return text
