"""Turkish punctuation normalization.

Cleans and normalizes punctuation for TTS input.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation for Turkish TTS.

    TODO:
        - Apostrophe normalization: Ankara'da, Türkiye'nin (keep for suffix boundary)
        - Quotation marks: "..." -> '' or spoken pause
        - Dash handling: em dash, en dash, hyphen in compounds
        - Ellipsis: ... -> spoken pause marker
        - Parentheses: (xyz) -> drop or read as pause
        - URL/email: skip or spell out
        - Hashtag/mention: skip
        - Whitespace normalization: multiple spaces, tabs, newlines
        - Unicode normalization: NFC for Turkish chars
    """
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # TODO: implement full punctuation normalization
    return text


def remove_non_speech(text: str) -> str:
    """Remove elements that have no spoken form (URLs, HTML tags, etc.).

    TODO:
        - Strip HTML/Markdown
        - Remove URLs
        - Remove email addresses
        - Remove @mentions and #hashtags
    """
    # Placeholder
    return text
