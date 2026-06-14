"""Gokbilge TTS command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="gokbilge-tts",
    help="Gokbilge TTS — Open Turkish Text-to-Speech Toolkit",
    no_args_is_help=True,
)
console = Console()


@app.command()
def normalize(
    text: str = typer.Argument(..., help="Turkish text to normalize"),
) -> None:
    """Normalize Turkish text (numbers, dates, abbreviations)."""
    from gokbilge_tts.normalize.text import normalize_text

    result = normalize_text(text)
    console.print(result)


@app.command()
def phonemize(
    text: str = typer.Argument(..., help="Turkish text to phonemize"),
) -> None:
    """Convert Turkish text to phoneme sequence."""
    from gokbilge_tts.normalize.text import normalize_text
    from gokbilge_tts.g2p.turkish import text_to_phonemes

    normalized = normalize_text(text)
    phonemes = text_to_phonemes(normalized)
    console.print(phonemes)


@app.command("prepare-issai")
def prepare_issai(
    dataset_dir: Path = typer.Option(..., "--dataset-dir", help="Path to local ISSAI dataset"),
    out: Path = typer.Option(..., "--out", help="Output directory for manifests"),
) -> None:
    """Prepare ISSAI Turkish Speech Corpus manifests for training."""
    from gokbilge_tts.datasets.prepare_issai import prepare

    console.print(f"Preparing ISSAI dataset from [bold]{dataset_dir}[/bold]...")
    prepare(dataset_dir=dataset_dir, output_dir=out)
    console.print(f"Manifests written to [bold]{out}[/bold]")


@app.command()
def infer(
    model: Path = typer.Option(..., "--model", help="Path to ONNX or VITS model"),
    text: str = typer.Option(..., "--text", help="Text to synthesize"),
    out: Path = typer.Option(Path("output.wav"), "--out", help="Output WAV path"),
    speaker_id: Optional[int] = typer.Option(None, "--speaker-id", help="Speaker ID (multi-speaker models)"),
) -> None:
    """Synthesize speech from text."""
    # TODO: route to piper_infer or vits_infer based on model type
    console.print(f"[yellow]Inference not yet implemented.[/yellow]")
    console.print(f"Model: {model}")
    console.print(f"Text:  {text!r}")
    console.print(f"Out:   {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
