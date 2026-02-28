"""CLI tool for converting PDFs to audiobooks."""

from __future__ import annotations

import logging
import os
import sys

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from .pdf_extract import extract_chapters, get_pdf_info
from .tts_engine import TTSEngine, DEFAULT_VOICE, detect_device, auto_detect_workers
from .model_manager import is_setup_complete, get_model_dir, list_local_voices
from .manifest import JobManifest

console = Console()


@click.group()
@click.version_option(package_name="pdf2audio")
def main():
    """pdf2audio - Convert PDFs to audiobooks using Kokoro TTS.

    Extracts chapters from PDF files and generates audio for each chapter,
    outputting files as they complete so you can start listening immediately.
    """
    pass


@main.command()
@click.option("--model-dir", default=None, help="Custom directory to store model files")
def setup(model_dir):
    """Download model files for local offline use (one-time setup).

    Downloads the Kokoro TTS model weights, config, and voice files
    to ~/.pdf2audio/models/ (or a custom directory). After this,
    everything runs fully locally with no internet required.
    """
    from .model_manager import download_models
    from pathlib import Path

    target_dir = Path(model_dir) if model_dir else get_model_dir()

    if is_setup_complete(target_dir):
        console.print(f"[green]Models already downloaded at {target_dir}[/green]")
        voices = list_local_voices(target_dir)
        console.print(f"  {len(voices)} voices available: {', '.join(voices[:5])}...")
        return

    console.print(Panel("[bold]pdf2audio setup[/bold] - Downloading model files", style="blue"))
    console.print(f"  Target: [cyan]{target_dir}[/cyan]")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=None)

        def on_progress(step, total, msg):
            progress.update(task, total=total, completed=step, description=msg)

        download_models(model_dir=target_dir, progress_callback=on_progress)

    console.print()
    voices = list_local_voices(target_dir)
    console.print(Panel(
        f"[bold green]Setup complete![/bold green]\n"
        f"Models saved to {target_dir}\n"
        f"{len(voices)} voices downloaded",
        style="green",
    ))


@main.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_dir", default=None, help="Output directory (default: ./output/<pdf_name>)")
@click.option("-v", "--voice", default=DEFAULT_VOICE, help=f"Kokoro voice name (default: {DEFAULT_VOICE})")
@click.option("-s", "--speed", default=1.0, type=float, help="Speech speed multiplier (default: 1.0)")
@click.option("-l", "--lang", "lang_code", default="a", help="Language code: a=US English, b=UK English (default: a)")
@click.option("-d", "--device", default=None, help="Compute device: cuda, cpu, mps (default: auto-detect)")
@click.option("-f", "--format", "output_format", default="wav", type=click.Choice(["wav", "mp3", "flac", "ogg"]), help="Audio format")
@click.option("-q", "--quality", default="medium", type=click.Choice(["low", "medium", "high"]), help="MP3 quality (default: medium)")
@click.option("-w", "--workers", default=None, type=int, help="Number of parallel workers (default: auto-detect)")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def convert(pdf_path, output_dir, voice, speed, lang_code, device, output_format, quality, workers, verbose):
    """Convert a PDF to audiobook files, one per chapter.

    Audio files are output as they are generated, so you can start
    listening to Chapter 1 while later chapters are still processing.

    Supports resume: if generation is interrupted, re-run the same
    command and already-completed chapters will be skipped.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Determine output dir
    if output_dir is None:
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join("output", basename)

    # Check setup
    if not is_setup_complete():
        console.print("[red]Models not downloaded yet.[/red]")
        console.print("Run [bold]pdf2audio setup[/bold] first to download model files.")
        sys.exit(1)

    console.print(Panel(f"[bold]pdf2audio[/bold] - PDF to Audiobook Converter", style="blue"))

    # Show device info
    dev = device or detect_device()
    num_workers = workers or auto_detect_workers(dev)
    console.print(f"  Device:  [cyan]{dev}[/cyan]")
    console.print(f"  Workers: [cyan]{num_workers}[/cyan]")
    console.print(f"  Voice:   [cyan]{voice}[/cyan]")
    console.print(f"  Speed:   [cyan]{speed}x[/cyan]")
    console.print(f"  Format:  [cyan]{output_format}[/cyan]")
    if output_format == "mp3":
        console.print(f"  Quality: [cyan]{quality}[/cyan]")
    console.print()

    # Extract chapters
    with console.status("[bold green]Extracting chapters from PDF..."):
        chapters = extract_chapters(pdf_path)

    if not chapters:
        console.print("[red]No text content found in PDF.[/red]")
        sys.exit(1)

    # Show chapter list
    table = Table(title=f"Found {len(chapters)} chapter(s)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", style="bold")
    table.add_column("Pages", justify="right")
    table.add_column("Characters", justify="right")

    for ch in chapters:
        table.add_row(
            str(ch.index + 1),
            ch.title[:60],
            f"{ch.page_start}-{ch.page_end}",
            f"{len(ch.text):,}",
        )
    console.print(table)
    console.print()

    # Set up manifest for resume support
    manifest = JobManifest(
        output_dir,
        pdf_path=pdf_path,
        voice=voice,
        speed=speed,
        lang_code=lang_code,
        format=output_format,
        quality=quality,
    )

    # Check if manifest matches; if settings changed, start fresh
    if manifest.manifest_path.exists() and not manifest.matches_settings(
        pdf_path, voice=voice, speed=speed, lang_code=lang_code, format=output_format, quality=quality
    ):
        console.print("[yellow]Settings changed from previous run. Starting fresh.[/yellow]")
        manifest = JobManifest(
            output_dir, pdf_path=pdf_path,
            voice=voice, speed=speed, lang_code=lang_code,
            format=output_format, quality=quality,
        )

    # Generate audio
    engine = TTSEngine(
        voice=voice,
        speed=speed,
        lang_code=lang_code,
        device=device,
        output_format=output_format,
        quality=quality,
        max_workers=workers,
    )

    console.print(f"Output directory: [cyan]{output_dir}[/cyan]")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        chapter_task = progress.add_task("Generating audio...", total=len(chapters))

        def on_progress(p):
            pct = p.chars_total > 0 and round(p.chars_processed / p.chars_total * 100) or 0
            progress.update(
                chapter_task,
                description=f"Ch {p.chapter_index + 1}: {p.chapter_title[:30]}... {pct}%",
            )

        for result in engine.generate_all(chapters, output_dir, manifest=manifest, progress_callback=on_progress):
            progress.update(chapter_task, advance=1, description="Generating audio...")
            if result.skipped:
                console.print(
                    f"  [blue]\u21B7[/blue] Chapter {result.chapter.index + 1}: "
                    f"[bold]{result.chapter.title}[/bold] "
                    f"({result.duration_seconds:.0f}s audio) "
                    f"[dim][skipped - cached][/dim]"
                )
            else:
                console.print(
                    f"  [green]\u2713[/green] Chapter {result.chapter.index + 1}: "
                    f"[bold]{result.chapter.title}[/bold] "
                    f"({result.duration_seconds:.0f}s audio, "
                    f"took {result.generation_time_seconds:.0f}s) "
                    f"-> [dim]{result.output_path}[/dim]"
                )

    console.print()
    console.print(Panel("[bold green]Done![/bold green] All chapters generated.", style="green"))


@main.command()
@click.argument("pdf_path", type=click.Path(exists=True))
def inspect(pdf_path):
    """Inspect a PDF file and show its chapters without generating audio."""
    info = get_pdf_info(pdf_path)
    console.print(Panel(f"[bold]{os.path.basename(pdf_path)}[/bold]", style="blue"))
    console.print(f"  Pages: {info['pages']}")
    console.print(f"  Has TOC: {'Yes' if info['has_toc'] else 'No'}")

    if info["metadata"]:
        meta = info["metadata"]
        if meta.get("title"):
            console.print(f"  Title: {meta['title']}")
        if meta.get("author"):
            console.print(f"  Author: {meta['author']}")

    console.print()

    with console.status("[bold green]Extracting chapters..."):
        chapters = extract_chapters(pdf_path)

    if not chapters:
        console.print("[yellow]No chapters detected.[/yellow]")
        return

    table = Table(title=f"{len(chapters)} chapter(s) found")
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", style="bold")
    table.add_column("Pages", justify="right")
    table.add_column("Characters", justify="right")
    table.add_column("Est. Audio", justify="right", style="cyan")

    for ch in chapters:
        # Rough estimate: ~150 words/min, ~5 chars/word at 1.0x speed
        est_minutes = len(ch.text) / (150 * 5)
        table.add_row(
            str(ch.index + 1),
            ch.title[:60],
            f"{ch.page_start}-{ch.page_end}",
            f"{len(ch.text):,}",
            f"~{est_minutes:.0f} min",
        )
    console.print(table)


@main.command()
def voices():
    """List available Kokoro TTS voices."""
    voice_list = [
        ("af_heart", "American English", "Female", "A"),
        ("af_bella", "American English", "Female", "A-"),
        ("af_nicole", "American English", "Female", "B-"),
        ("af_aoede", "American English", "Female", "B-"),
        ("af_kore", "American English", "Female", "B-"),
        ("af_sarah", "American English", "Female", "B"),
        ("af_sky", "American English", "Female", "B-"),
        ("am_adam", "American English", "Male", "B"),
        ("am_michael", "American English", "Male", "B"),
        ("am_fenrir", "American English", "Male", "B-"),
        ("bf_emma", "British English", "Female", "B-"),
        ("bf_isabella", "British English", "Female", "C+"),
        ("bm_george", "British English", "Male", "C+"),
        ("bm_lewis", "British English", "Male", "C+"),
        ("ff_siwis", "French", "Female", "B-"),
        ("jf_alpha", "Japanese", "Female", "C+"),
    ]

    table = Table(title="Available Voices")
    table.add_column("Voice ID", style="cyan bold")
    table.add_column("Language")
    table.add_column("Gender")
    table.add_column("Quality", justify="center")

    for vid, lang, gender, quality in voice_list:
        style = "green" if quality.startswith("A") else "yellow" if quality.startswith("B") else "dim"
        table.add_row(vid, lang, gender, f"[{style}]{quality}[/{style}]")

    console.print(table)
    console.print("\n[dim]Use --voice <voice_id> with the convert command.[/dim]")


@main.command()
def devices():
    """Show available compute devices."""
    import torch

    console.print("[bold]Compute Device Detection[/bold]\n")

    if torch.cuda.is_available():
        console.print(f"  [green]\u2713[/green] CUDA: {torch.cuda.get_device_name(0)}")
        console.print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        console.print("  [red]\u2717[/red] CUDA: Not available")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        console.print("  [green]\u2713[/green] MPS: Apple Silicon available")
    else:
        console.print("  [red]\u2717[/red] MPS: Not available")

    console.print("  [green]\u2713[/green] CPU: Always available")
    console.print()

    best = detect_device()
    console.print(f"  Auto-detected: [bold cyan]{best}[/bold cyan]")


@main.command()
@click.option("-h", "--host", default="127.0.0.1", help="Host to bind to")
@click.option("-p", "--port", default=5000, type=int, help="Port to listen on")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def web(host, port, debug):
    """Start the web interface."""
    console.print(Panel("[bold]pdf2audio[/bold] Web Interface", style="blue"))
    console.print(f"  Starting server at [cyan]http://{host}:{port}[/cyan]")
    console.print("  Press Ctrl+C to stop\n")

    from .web.app import create_app
    create_app(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
