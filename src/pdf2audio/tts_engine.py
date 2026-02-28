"""TTS engine wrapping Kokoro for chapter-by-chapter audio generation."""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf
import torch

from .pdf_extract import Chapter

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

# Voices rated A or B by Kokoro
DEFAULT_VOICES = {
    "a": "af_heart",       # American English female (A rating)
    "b": "bf_emma",        # British English female (B-)
    "f": "ff_siwis",       # French female (B-)
    "j": "jf_alpha",       # Japanese female (C+)
}
DEFAULT_VOICE = "af_heart"


@dataclass
class AudioResult:
    """Result of generating audio for a chapter."""

    chapter: Chapter
    output_path: str
    duration_seconds: float
    generation_time_seconds: float


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
    ):
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self.device = device or detect_device()
        self.output_format = output_format
        self._pipeline = None

    def _ensure_pipeline(self):
        """Lazy-init the Kokoro pipeline."""
        if self._pipeline is None:
            from kokoro import KPipeline
            logger.info(f"Initializing Kokoro pipeline (lang={self.lang_code}, device={self.device})")
            self._pipeline = KPipeline(lang_code=self.lang_code, device=self.device)
            logger.info("Kokoro pipeline ready")

    def generate_chapter(
        self,
        chapter: Chapter,
        output_dir: str,
    ) -> AudioResult:
        """Generate audio for a single chapter, saving to disk.

        Returns the result as soon as the file is written.
        """
        self._ensure_pipeline()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build filename from chapter index and title
        safe_title = _safe_filename(chapter.title)
        filename = f"{chapter.index:03d}_{safe_title}.{self.output_format}"
        output_path = output_dir / filename

        logger.info(f"Generating chapter {chapter.index}: {chapter.title} ({len(chapter.text)} chars)")
        start_time = time.time()

        # Generate audio chunks and concatenate
        all_audio: list[np.ndarray] = []
        chunk_count = 0

        for _gs, _ps, audio in self._pipeline(
            chapter.text,
            voice=self.voice,
            speed=self.speed,
            split_pattern=r'\n+',
        ):
            if audio is not None and len(audio) > 0:
                all_audio.append(audio)
                chunk_count += 1

        if not all_audio:
            logger.warning(f"No audio generated for chapter {chapter.index}: {chapter.title}")
            # Write a short silence so the file exists
            all_audio = [np.zeros(SAMPLE_RATE, dtype=np.float32)]

        combined = np.concatenate(all_audio)
        duration = len(combined) / SAMPLE_RATE
        gen_time = time.time() - start_time

        sf.write(str(output_path), combined, SAMPLE_RATE)
        logger.info(
            f"Chapter {chapter.index} done: {duration:.1f}s audio, "
            f"generated in {gen_time:.1f}s ({chunk_count} chunks)"
        )

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
    ) -> Generator[AudioResult, None, None]:
        """Generate audio for all chapters, yielding each result as it completes.

        This is the key streaming API - callers get each chapter's audio file
        as soon as it's ready, so playback can start immediately.
        """
        total = len(chapters)
        for i, chapter in enumerate(chapters):
            logger.info(f"Processing chapter {i + 1}/{total}")
            result = self.generate_chapter(chapter, output_dir)
            yield result


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
