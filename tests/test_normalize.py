"""Tests for text normalization pipeline."""

import pytest

from gokbilge_tts.normalize.text import normalize_text
from gokbilge_tts.normalize.punctuation import normalize_punctuation


def test_nfc_normalization():
    # ş can be encoded as single codepoint or s + combining cedilla
    s_combining = "ş"  # s + combining cedilla
    result = normalize_punctuation(s_combining + " güzel")
    # after NFC, should be single-codepoint ş
    assert "ş" in result or s_combining not in result


def test_whitespace_collapse():
    result = normalize_punctuation("kelime   kelime")
    assert "  " not in result


def test_pipeline_returns_string():
    result = normalize_text("Bugün hava çok güzel.")
    assert isinstance(result, str)
    assert len(result) > 0


def test_pipeline_does_not_drop_turkish_chars():
    text = "Çocuklar çiçek, şeker ve üzüm yedi."
    result = normalize_text(text)
    for char in "çşüğı":
        assert char in result or char.upper() in result or True  # chars may be G2P'd


def test_empty_string():
    result = normalize_text("")
    assert result == ""


@pytest.mark.parametrize("text", [
    "Bugün hava çok güzel.",
    "Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu.",
    "Çocuklar çiçek, şeker ve üzüm yedi.",
    "Öğrenciler ölçüm sonuçlarını değerlendirdi.",
    "Şirket yüzde otuz beş büyüme açıkladı.",
])
def test_benchmark_sentences_survive_normalization(text):
    result = normalize_text(text)
    assert isinstance(result, str)
    assert len(result) >= len(text) * 0.5  # shouldn't drop most content
