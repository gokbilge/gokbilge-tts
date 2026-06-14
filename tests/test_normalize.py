"""Tests for text normalization pipeline."""

import pytest

from gokbilge_tts.normalize.text import normalize_text
from gokbilge_tts.normalize.punctuation import normalize_punctuation, remove_non_speech
from gokbilge_tts.normalize.numbers import _int_to_words, _int_to_ordinal, normalize_numbers
from gokbilge_tts.normalize.abbreviations import normalize_abbreviations


# ---------------------------------------------------------------------------
# Punctuation
# ---------------------------------------------------------------------------

def test_nfc_normalization():
    # ş can be s + combining cedilla; NFC must collapse it to a single codepoint
    s_combining = "ş"  # s + combining cedilla (not single-codepoint ş)
    result = normalize_punctuation(s_combining + " güzel")
    assert "ş" in result  # single-codepoint ş


def test_whitespace_collapse():
    result = normalize_punctuation("kelime   kelime")
    assert "  " not in result


def test_remove_url():
    result = remove_non_speech("Siteye https://example.com bakın.")
    assert "https" not in result


def test_remove_html():
    result = remove_non_speech("<b>Merhaba</b> dünya")
    assert "<b>" not in result
    assert "Merhaba" in result


# ---------------------------------------------------------------------------
# Pipeline basics
# ---------------------------------------------------------------------------

def test_pipeline_returns_string():
    result = normalize_text("Bugün hava çok güzel.")
    assert isinstance(result, str)
    assert len(result) > 0


def test_pipeline_does_not_drop_turkish_chars():
    text = "Çocuklar çiçek, şeker ve üzüm yedi."
    result = normalize_text(text)
    # ç, ş, ü appear in input — they must survive normalization unchanged
    assert "ç" in result or "Ç" in result
    assert "ş" in result or "Ş" in result
    assert "ü" in result or "Ü" in result


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
    assert len(result) >= len(text) * 0.5


# ---------------------------------------------------------------------------
# Number normalization — _int_to_words
# ---------------------------------------------------------------------------

def test_zero():
    assert _int_to_words(0) == "sıfır"


def test_ones():
    assert _int_to_words(1) == "bir"
    assert _int_to_words(9) == "dokuz"


def test_tens():
    assert _int_to_words(10) == "on"
    assert _int_to_words(20) == "yirmi"
    assert _int_to_words(99) == "doksan dokuz"


def test_hundreds():
    assert _int_to_words(100) == "yüz"       # "yüz" not "bir yüz"
    assert _int_to_words(200) == "iki yüz"
    assert _int_to_words(999) == "dokuz yüz doksan dokuz"


def test_thousands():
    assert _int_to_words(1000) == "bin"       # "bin" not "bir bin"
    assert _int_to_words(2000) == "iki bin"
    assert _int_to_words(1923) == "bin dokuz yüz yirmi üç"


def test_millions():
    assert _int_to_words(1_000_000) == "bir milyon"
    assert _int_to_words(2_500_000) == "iki milyon beş yüz bin"


def test_negative():
    assert _int_to_words(-5) == "eksi beş"


# ---------------------------------------------------------------------------
# Number normalization — ordinals
# ---------------------------------------------------------------------------

def test_ordinal_birinci():
    assert _int_to_ordinal(1) == "birinci"


def test_ordinal_ucuncu():
    assert _int_to_ordinal(3) == "üçüncü"


def test_ordinal_onuncu():
    assert _int_to_ordinal(10) == "onuncu"


def test_ordinal_bininci():
    assert _int_to_ordinal(1000) == "bininci"


# ---------------------------------------------------------------------------
# Number normalization — normalize_numbers patterns
# ---------------------------------------------------------------------------

def test_percentage():
    assert normalize_numbers("%35") == "yüzde otuz beş"
    assert normalize_numbers("% 35") == "yüzde otuz beş"


def test_decimal_comma():
    result = normalize_numbers("3,14")
    assert "üç" in result
    assert "virgül" in result
    assert "on dört" in result


def test_thousands_separator():
    assert normalize_numbers("1.000.000") == "bir milyon"
    assert normalize_numbers("1.000") == "bin"


def test_plain_integer_1923():
    assert normalize_numbers("1923") == "bin dokuz yüz yirmi üç"


def test_number_in_sentence():
    result = normalize_numbers("Şirket %35 büyüdü.")
    assert "yüzde otuz beş" in result


def test_tl_currency_pipeline():
    # Full pipeline: 50 TL -> abbrev expands TL, then numbers converts 50
    result = normalize_text("50 TL")
    assert "elli" in result
    assert "Türk lirası" in result


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------

def test_abbrev_dr():
    assert "doktor" in normalize_abbreviations("Dr. Fatih")


def test_abbrev_prof():
    result = normalize_abbreviations("Prof. Dr. Ahmet")
    assert "profesör" in result
    assert "doktor" in result


def test_abbrev_sn():
    assert "sayın" in normalize_abbreviations("Sn. Yılmaz")


def test_abbrev_no():
    assert "numara" in normalize_abbreviations("No. 5")


def test_abbrev_tl_standalone():
    result = normalize_abbreviations("50 TL")
    assert "Türk lirası" in result
    assert "TL" not in result


def test_abbrev_tl_not_inside_word():
    # TL should not match inside a longer word
    result = normalize_abbreviations("ATLAS")
    assert result == "ATLAS"


def test_abbrev_km():
    assert "kilometre" in normalize_abbreviations("5 km")
