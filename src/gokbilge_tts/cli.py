"""Gokbilge TTS command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

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
    dataset_dir: Optional[Path] = typer.Option(
        None, "--dataset-dir",
        help="Local ISSAI directory (Train/Dev/Test subdirs). Omit to load from HuggingFace.",
    ),
    out: Path = typer.Option(..., "--out", help="Output directory for manifests"),
) -> None:
    """Prepare ISSAI Turkish Speech Corpus manifests for training."""
    from gokbilge_tts.datasets.prepare_issai import prepare

    src = str(dataset_dir) if dataset_dir else "HuggingFace"
    console.print(f"Preparing ISSAI from [bold]{src}[/bold]…")
    prepare(dataset_dir=dataset_dir, output_dir=out)
    console.print(f"Manifests written to [bold]{out}[/bold]")


@app.command("validate-manifest")
def validate_manifest_cmd(
    manifest: Path = typer.Argument(..., help="Path to JSONL manifest file to validate"),
    skip_audio: bool = typer.Option(False, "--skip-audio", help="Skip audio file existence check"),
    min_duration: float = typer.Option(0.5, "--min-duration", help="Minimum duration in seconds"),
    max_duration: float = typer.Option(20.0, "--max-duration", help="Maximum duration in seconds"),
) -> None:
    """Validate a JSONL manifest (fields, audio existence, duration, phonemes)."""
    from gokbilge_tts.datasets.validate_manifest import validate_manifest

    console.print(f"Validating [bold]{manifest}[/bold] …")
    valid, errors = validate_manifest(
        manifest,
        check_audio=not skip_audio,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    for err in errors:
        console.print(f"  [red]ERROR:[/red] {err}")

    status = "[green]PASS[/green]" if not errors else "[red]FAIL[/red]"
    console.print(f"{status}  valid={valid}  errors={len(errors)}")

    if errors:
        raise typer.Exit(code=1)


@app.command("export-piper")
def export_piper_cmd(
    manifest_dir: Path = typer.Option(..., "--manifest-dir", help="Directory with train/val.jsonl and symbols.txt"),
    out: Path = typer.Option(..., "--out", help="Output directory for Piper LJSpeech files"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Cap training records for smoke testing"),
    sample_rate: int = typer.Option(22050, "--sample-rate", help="Target audio sample rate Hz"),
    language: str = typer.Option("tr", "--language", help="espeak-ng language code for config.json"),
) -> None:
    """Convert Gokbilge manifests to Piper LJSpeech training format.

    Writes wavs/ (symlinks), metadata.csv, config.json, and split .txt files
    to --out. Run piper_train.preprocess on --out next.
    """
    from gokbilge_tts.datasets.export_piper import export_piper

    console.print(f"Exporting Piper dataset: [bold]{manifest_dir}[/bold] → [bold]{out}[/bold]")
    if limit is not None:
        console.print(f"  [yellow]Smoke mode:[/yellow] limiting training to {limit} records")

    export_piper(
        manifest_dir=manifest_dir,
        out_dir=out,
        limit=limit,
        sample_rate=sample_rate,
        language=language,
    )
    console.print(f"[green]Done.[/green] Piper dataset at [bold]{out}[/bold]")


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
