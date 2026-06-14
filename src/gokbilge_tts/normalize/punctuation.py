"""Turkish punctuation normalization.

Cleans and normalizes punctuation for TTS input.
"""

from __future__ import annotations

import re
import unicodedata

# Compiled patterns
_RE_HTML = re.compile(r'<[^>]+>')
_RE_URL = re.compile(r'https?://\S+|www\.\S+')
_RE_EMAIL = re.compile(r'\S+@\S+\.[a-zA-Z]{2,}')
_RE_MENTION = re.compile(r'@\w+')
_RE_HASHTAG = re.compile(r'#\w+')
_RE_WHITESPACE = re.compile(r'\s+')

# Typographic quote pairs -> plain ASCII equivalents
_QUOTE_MAP = str.maketrans({
    '“': '"',  # "
    '”': '"',  # "
    '‘': "'",  # '
    '’': "'",  # '
    '«': '"',  # «
    '»': '"',  # »
    '„': '"',  # „
})

# Em dash / en dash -> comma (spoken pause in Turkish)
_DASH_MAP = str.maketrans({
    '—': ',',  # —
    '–': ',',  # –
})


def remove_non_speech(text: str) -> str:
    """Remove elements that have no spoken form: HTML tags, URLs, emails, mentions, hashtags."""
    text = _RE_HTML.sub(' ', text)
    text = _RE_URL.sub(' ', text)
    text = _RE_EMAIL.sub(' ', text)
    text = _RE_MENTION.sub(' ', text)
    text = _RE_HASHTAG.sub(' ', text)
    return text


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation for Turkish TTS.

    Steps:
    - Unicode NFC (ensures single-codepoint Turkish characters)
    - Typographic quote -> plain quote
    - Em/en dash -> comma
    - Parenthesized content: strip parens, keep content
    - Collapse whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_QUOTE_MAP)
    text = text.translate(_DASH_MAP)

    # Remove parentheses but keep the enclosed content (may have spoken value)
    text = text.replace('(', ' ').replace(')', ' ')
    text = text.replace('[', ' ').replace(']', ' ')

    # Collapse whitespace
    text = _RE_WHITESPACE.sub(' ', text).strip()

    return text
