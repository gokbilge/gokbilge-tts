"""Optional phonemizer/espeak-ng backend for comparison.

This module provides an optional interface to the `phonemizer` library
(which uses espeak-ng under the hood) for Turkish G2P output comparison.

It is NOT the primary G2P path. Use gokbilge_tts.g2p.turkish for the
built-in rule-based system.

Install optional dependency:
    pip install "gokbilge-tts[phonemizer]"
    # also requires espeak-ng system package:
    # apt-get install espeak-ng
"""

from __future__ import annotations


def phonemize_with_espeak(text: str, language: str = "tr") -> str:
    """Phonemize Turkish text using espeak-ng via phonemizer library.

    Args:
        text: Normalized Turkish text.
        language: espeak-ng language code (default: "tr").

    Returns:
        IPA phoneme string from espeak-ng.

    Raises:
        ImportError: if phonemizer is not installed.
        RuntimeError: if espeak-ng is not available on the system.

    TODO:
        - Compare output with built-in turkish.py G2P
        - Use for benchmarking only, not as primary path
        - Handle punctuation stripping before phonemizing
    """
    try:
        from phonemizer import phonemize  # type: ignore
    except ImportError:
        raise ImportError(
            "phonemizer is not installed. "
            "Install with: pip install 'gokbilge-tts[phonemizer]'"
        )

    result: str = phonemize(
        text,
        backend="espeak",
        language=language,
        with_stress=True,
        preserve_punctuation=False,
    )
    return result
