"""CLI smoke tests. Requires typer (pip install 'gokbilge-tts[dev]')."""

import pytest

typer_testing = pytest.importorskip("typer.testing", reason="typer not installed")
CliRunner = typer_testing.CliRunner

from gokbilge_tts.cli import app  # noqa: E402


runner = CliRunner()


def test_normalize_command_basic():
    result = runner.invoke(app, ["normalize", "Bugün hava güzel."])
    assert result.exit_code == 0
    assert len(result.output.strip()) > 0


def test_normalize_converts_percentage():
    result = runner.invoke(app, ["normalize", "Şirket %35 büyüdü."])
    assert result.exit_code == 0
    assert "yüzde otuz beş" in result.output


def test_normalize_converts_tl():
    result = runner.invoke(app, ["normalize", "50 TL ödedi."])
    assert result.exit_code == 0
    assert "elli" in result.output
    assert "Türk lirası" in result.output


def test_phonemize_command_basic():
    result = runner.invoke(app, ["phonemize", "güzel"])
    assert result.exit_code == 0
    output = result.output
    # Should contain IPA phonemes for g-ü-z-e-l
    assert "y" in output    # ü -> y
    assert "z" in output


def test_phonemize_dotless_i():
    result = runner.invoke(app, ["phonemize", "ılık"])
    assert result.exit_code == 0
    assert "ɯ" in result.output


def test_normalize_empty_string():
    result = runner.invoke(app, ["normalize", ""])
    assert result.exit_code == 0


def test_help_flag():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "normalize" in result.output
    assert "phonemize" in result.output
