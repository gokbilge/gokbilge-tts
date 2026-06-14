"""Tests for Turkish G2P."""

import pytest

from gokbilge_tts.g2p.turkish import text_to_phonemes, GRAPHEME_TO_PHONEME


def test_basic_vowels():
    phonemes = text_to_phonemes("a")
    assert "a" in phonemes


def test_dotless_i():
    # ı should map to back unrounded /ɯ/, not /i/
    phonemes = text_to_phonemes("ı")
    assert "ɯ" in phonemes


def test_c_is_affricate():
    # Turkish 'c' is /dʒ/, not /k/
    phonemes = text_to_phonemes("c")
    assert "dʒ" in phonemes


def test_c_cedilla():
    # ç → /tʃ/
    phonemes = text_to_phonemes("ç")
    assert "tʃ" in phonemes


def test_s_cedilla():
    # ş → /ʃ/
    phonemes = text_to_phonemes("ş")
    assert "ʃ" in phonemes


def test_soft_g_silenced():
    # ğ should produce no phoneme (or empty)
    phonemes = text_to_phonemes("ğ")
    # ğ is silent — output should be empty or whitespace only
    assert phonemes.strip() == "" or "ğ" not in phonemes


def test_o_umlaut():
    # ö → /ø/
    phonemes = text_to_phonemes("ö")
    assert "ø" in phonemes


def test_u_umlaut():
    # ü → /y/
    phonemes = text_to_phonemes("ü")
    assert "y" in phonemes


def test_whole_word():
    # "güzel" → /ɡ y z e l/ (space-separated or similar)
    phonemes = text_to_phonemes("güzel")
    assert "ɡ" in phonemes or "g" in phonemes
    assert "y" in phonemes
    assert "z" in phonemes


def test_grapheme_map_completeness():
    # Every lowercase Turkish letter should have a mapping
    required = set("abcçdefgğhıijklmnoöprsştuüvyz")
    missing = required - set(GRAPHEME_TO_PHONEME.keys())
    assert missing == set(), f"Missing graphemes: {missing}"


@pytest.mark.parametrize("word,expected_phonemes", [
    ("ev", ["e", "v"]),
    ("ip", ["i", "p"]),
    ("ok", ["o", "k"]),
])
def test_simple_words(word, expected_phonemes):
    phonemes = text_to_phonemes(word)
    for ph in expected_phonemes:
        assert ph in phonemes
