"""TTS engine wrapping Kokoro for chapter-by-chapter audio generation."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator

import numpy as np
import torch

from .pdf_extract import Chapter
from .model_manager import get_local_paths, get_voice_path, is_setup_complete
from .audio_formats import save_audio, SAMPLE_RATE
from .manifest import JobManifest

logger = logging.getLogger(__name__)

# Voices rated A or B by Kokoro
DEFAULT_VOICES = {
    "a": "af_heart",       # American English female (A rating)
    "b": "bf_emma",        # British English female (B-)
    "f": "ff_siwis",       # French female (B-)
    "j": "jf_alpha",       # Japanese female (C+)
}
DEFAULT_VOICE = "af_heart"

# Maximum characters per text segment for chunking long chapters
MAX_SEGMENT_CHARS = 2500
SEGMENT_SILENCE_SECONDS = 0.3


@dataclass
class AudioResult:
    """Result of generating audio for a chapter."""

    chapter: Chapter
    output_path: str
    duration_seconds: float
    generation_time_seconds: float
    skipped: bool = False  # True if loaded from checkpoint


@dataclass
class ChapterProgress:
    """Progress update for a chapter being generated."""

    chapter_index: int
    chapter_title: str
    chars_processed: int
    chars_total: int


# Type alias for progress callback
ProgressCallback = Callable[[ChapterProgress], None]


def detect_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA GPU: {gpu_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Silicon MPS")
    else:
        device = "cpu"
        logger.info("Using CPU (no GPU detected)")
    return device


class TTSEngine:
    """Kokoro TTS engine with streaming chapter generation."""

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        lang_code: str = "a",
        device: str | None = None,
        output_format: str = "wav",
        quality: str = "medium",
    ):
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self.device = device or detect_device()
        self.output_format = output_format
        self.quality = quality
        self._pipeline = None

    def _ensure_pipeline(self):
        """Lazy-init the Kokoro pipeline using local model files."""
        if self._pipeline is None:
            if not is_setup_complete():
                raise RuntimeError(
                    "Models not downloaded yet. Run 'pdf2audio setup' first to download "
                    "model files for offline use."
                )

            from kokoro import KPipeline, KModel

            paths = get_local_paths()
            logger.info(f"Loading model from local files: {paths['config']}")

            # Load model from local files — no HuggingFace downloads
            model = KModel(config=paths["config"], model=paths["model"])
            model = model.to(self.device).eval()

            self._pipeline = KPipeline(
                lang_code=self.lang_code,
                model=model,
                device=self.device,
            )

            # Resolve voice name to local .pt path so pipeline doesn't hit HF
            self._voice_path = get_voice_path(self.voice)
            logger.info(f"Kokoro pipeline ready (local, device={self.device})")

    def generate_chapter(
        self,
        chapter: Chapter,
        output_dir: str,
        progress_callback: ProgressCallback | None = None,
    ) -> AudioResult:
        """Generate audio for a single chapter, saving to disk.

        Long chapters are automatically split into segments at sentence
        boundaries and concatenated with brief silence gaps.
        """
        self._ensure_pipeline()

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Build filename from chapter index and title
        safe_title = _safe_filename(chapter.title)
        filename = f"{chapter.index:03d}_{safe_title}.{self.output_format}"
        output_path = output_dir_path / filename

        logger.info(f"Generating chapter {chapter.index}: {chapter.title} ({len(chapter.text)} chars)")
        start_time = time.time()

        # Split long chapters into manageable segments
        segments = _split_long_text(chapter.text, MAX_SEGMENT_CHARS)
        total_chars = len(chapter.text)
        chars_processed = 0

        all_audio: list[np.ndarray] = []
        silence_gap = np.zeros(int(SEGMENT_SILENCE_SECONDS * SAMPLE_RATE), dtype=np.float32)

        for seg_idx, segment_text in enumerate(segments):
            # Generate audio for this segment
            for gs, _ps, audio in self._pipeline(
                segment_text,
                voice=self._voice_path,
                speed=self.speed,
                split_pattern=r'\n+',
            ):
                if audio is not None and len(audio) > 0:
                    all_audio.append(audio)

                # Track progress using graphemes length
                if gs:
                    chars_processed += len(gs)
                    if progress_callback:
                        progress_callback(ChapterProgress(
                            chapter_index=chapter.index,
                            chapter_title=chapter.title,
                            chars_processed=min(chars_processed, total_chars),
                            chars_total=total_chars,
                        ))

            # Add silence between segments (but not after the last one)
            if seg_idx < len(segments) - 1 and all_audio:
                all_audio.append(silence_gap)

        if not all_audio:
            logger.warning(f"No audio generated for chapter {chapter.index}: {chapter.title}")
            all_audio = [np.zeros(SAMPLE_RATE, dtype=np.float32)]

        combined = np.concatenate(all_audio)
        duration = len(combined) / SAMPLE_RATE
        gen_time = time.time() - start_time

        save_audio(combined, str(output_path), SAMPLE_RATE, self.output_format, self.quality)
        logger.info(
            f"Chapter {chapter.index} done: {duration:.1f}s audio, "
            f"generated in {gen_time:.1f}s ({len(segments)} segments)"
        )

        # Final progress update
        if progress_callback:
            progress_callback(ChapterProgress(
                chapter_index=chapter.index,
                chapter_title=chapter.title,
                chars_processed=total_chars,
                chars_total=total_chars,
            ))

        return AudioResult(
            chapter=chapter,
            output_path=str(output_path),
            duration_seconds=duration,
            generation_time_seconds=gen_time,
        )

    def generate_all(
        self,
        chapters: list[Chapter],
        output_dir: str,
        manifest: JobManifest | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Generator[AudioResult, None, None]:
        """Generate audio for all chapters, yielding each result as it completes.

        If a manifest is provided, already-completed chapters are skipped
        and their cached results are yielded immediately.
        """
        total = len(chapters)
        for i, chapter in enumerate(chapters):
            # Check if already done (resume support)
            if manifest and manifest.is_chapter_done(chapter.index):
                cached = manifest.get_chapter_result(chapter.index)
                logger.info(f"Skipping chapter {i + 1}/{total} (cached): {chapter.title}")
                yield AudioResult(
                    chapter=chapter,
                    output_path=cached["output_path"],
                    duration_seconds=cached["duration_seconds"],
                    generation_time_seconds=0,
                    skipped=True,
                )
                continue

            logger.info(f"Processing chapter {i + 1}/{total}")
            result = self.generate_chapter(chapter, output_dir, progress_callback)

            # Save to manifest after successful generation
            if manifest:
                manifest.mark_chapter_done(chapter.index, result.output_path, result.duration_seconds)

            yield result


def _split_long_text(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split text into segments at sentence boundaries.

    Returns the original text as a single-element list if it's short enough.
    """
    if len(text) <= max_chars:
        return [text]

    segments = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            segments.append(remaining)
            break

        # Find the best split point within max_chars
        chunk = remaining[:max_chars]

        # Try sentence-ending punctuation first (search from the end)
        split_pos = -1
        for pattern in ['. ', '? ', '! ', '.\n', '?\n', '!\n']:
            pos = chunk.rfind(pattern)
            if pos > max_chars // 4:  # Don't split too early
                split_pos = pos + len(pattern)
                break

        # Fall back to newline
        if split_pos == -1:
            pos = chunk.rfind('\n')
            if pos > max_chars // 4:
                split_pos = pos + 1

        # Fall back to any space
        if split_pos == -1:
            pos = chunk.rfind(' ')
            if pos > max_chars // 4:
                split_pos = pos + 1

        # Last resort: hard split
        if split_pos == -1:
            split_pos = max_chars

        segments.append(remaining[:split_pos].rstrip())
        remaining = remaining[split_pos:].lstrip()

    return [s for s in segments if s.strip()]


def _safe_filename(title: str, max_len: int = 60) -> str:
    """Convert a chapter title to a safe filename."""
    # Replace unsafe chars
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    # Collapse runs of underscores/spaces
    safe = "_".join(safe.split())
    # Truncate
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("_")
    return safe or "untitled"
