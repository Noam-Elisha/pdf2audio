"""TTS engine wrapping Kokoro for chapter-by-chapter audio generation."""

from __future__ import annotations

import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class GenerationStopped(Exception):
    """Raised when generation is stopped by user."""
    pass


class JobControl:
    """Thread-safe controls for pausing/stopping a running job."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start in running (unpaused) state

    def stop(self):
        """Signal all workers to stop."""
        self._stop_event.set()
        self._pause_event.set()  # Unblock any paused workers so they can exit

    def pause(self):
        """Pause all workers (they block at next checkpoint)."""
        self._pause_event.clear()

    def resume(self):
        """Resume all paused workers."""
        self._pause_event.set()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    @property
    def paused(self) -> bool:
        return not self._pause_event.is_set()

    def check(self) -> bool:
        """Call between work units. Blocks if paused. Returns False if stopped."""
        self._pause_event.wait()
        return not self._stop_event.is_set()


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


def auto_detect_workers(device: str) -> int:
    """Auto-detect optimal number of parallel workers based on device."""
    if device == "cuda":
        # GPU: 2 workers to overlap I/O with compute (CUDA serializes anyway)
        return 2
    elif device == "mps":
        # MPS doesn't handle concurrent access well
        return 1
    else:
        # CPU: use half the cores, max 4
        cores = os.cpu_count() or 1
        return max(1, min(cores // 2, 4))


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
        max_workers: int | None = None,
    ):
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self.device = device or detect_device()
        self.output_format = output_format
        self.quality = quality
        self.max_workers = max_workers  # None = auto-detect
        self._model = None
        self._pipeline = None
        self._voice_path = None

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

            self._model = KModel(config=paths["config"], model=paths["model"])
            self._model = self._model.to(self.device).eval()

            self._pipeline = KPipeline(
                lang_code=self.lang_code,
                model=self._model,
                device=self.device,
            )

            self._voice_path = get_voice_path(self.voice)
            logger.info(f"Kokoro pipeline ready (local, device={self.device})")

    def _create_worker_pipeline(self):
        """Create a new KPipeline sharing the already-loaded model.

        Used for parallel workers — each thread gets its own pipeline
        (which has internal text-processing state) but they share the
        same model weights to save memory.
        """
        from kokoro import KPipeline

        pipeline = KPipeline(
            lang_code=self.lang_code,
            model=self._model,
            device=self.device,
        )
        return pipeline, self._voice_path

    def generate_chapter(
        self,
        chapter: Chapter,
        output_dir: str,
        progress_callback: ProgressCallback | None = None,
        control: JobControl | None = None,
    ) -> AudioResult:
        """Generate audio for a single chapter, saving to disk."""
        self._ensure_pipeline()
        return self._do_generate_chapter(
            chapter, output_dir, self._pipeline, self._voice_path,
            progress_callback, control,
        )

    def _do_generate_chapter(
        self,
        chapter: Chapter,
        output_dir: str,
        pipeline,
        voice_path: str,
        progress_callback: ProgressCallback | None,
        control: JobControl | None,
    ) -> AudioResult:
        """Internal: generate a chapter using a specific pipeline instance."""
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        safe_title = _safe_filename(chapter.title)
        filename = f"{chapter.index:03d}_{safe_title}.{self.output_format}"
        output_path = output_dir_path / filename

        logger.info(f"Generating chapter {chapter.index}: {chapter.title} ({len(chapter.text)} chars)")
        start_time = time.time()

        segments = _split_long_text(chapter.text, MAX_SEGMENT_CHARS)
        total_chars = len(chapter.text)
        chars_processed = 0

        all_audio: list[np.ndarray] = []
        silence_gap = np.zeros(int(SEGMENT_SILENCE_SECONDS * SAMPLE_RATE), dtype=np.float32)

        for seg_idx, segment_text in enumerate(segments):
            # Check control between segments
            if control and not control.check():
                raise GenerationStopped()

            for gs, _ps, audio in pipeline(
                segment_text,
                voice=voice_path,
                speed=self.speed,
                split_pattern=r'\n+',
            ):
                if audio is not None and len(audio) > 0:
                    all_audio.append(audio)

                if gs:
                    chars_processed += len(gs)
                    if progress_callback:
                        progress_callback(ChapterProgress(
                            chapter_index=chapter.index,
                            chapter_title=chapter.title,
                            chars_processed=min(chars_processed, total_chars),
                            chars_total=total_chars,
                        ))

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
        control: JobControl | None = None,
    ) -> Generator[AudioResult, None, None]:
        """Generate audio for all chapters, yielding each result as it completes.

        Automatically parallelizes based on device unless max_workers is set.
        If a manifest is provided, already-completed chapters are skipped.
        If a control is provided, generation can be paused/stopped.
        """
        workers = self.max_workers or auto_detect_workers(self.device)
        workers = min(workers, len(chapters))

        if workers <= 1:
            yield from self._generate_sequential(
                chapters, output_dir, manifest, progress_callback, control,
            )
        else:
            yield from self._generate_parallel(
                chapters, output_dir, manifest, progress_callback, control, workers,
            )

    def _generate_sequential(
        self,
        chapters: list[Chapter],
        output_dir: str,
        manifest: JobManifest | None,
        progress_callback: ProgressCallback | None,
        control: JobControl | None,
    ) -> Generator[AudioResult, None, None]:
        """Sequential generation — one chapter at a time."""
        total = len(chapters)
        for i, chapter in enumerate(chapters):
            # Check stop/pause between chapters
            if control and not control.check():
                break

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
            result = self.generate_chapter(chapter, output_dir, progress_callback, control)

            if manifest:
                manifest.mark_chapter_done(chapter.index, result.output_path, result.duration_seconds)

            yield result

    def _generate_parallel(
        self,
        chapters: list[Chapter],
        output_dir: str,
        manifest: JobManifest | None,
        progress_callback: ProgressCallback | None,
        control: JobControl | None,
        num_workers: int,
    ) -> Generator[AudioResult, None, None]:
        """Parallel generation — multiple chapters at once using a thread pool."""
        # Yield cached chapters first, collect pending ones
        pending = []
        for chapter in chapters:
            if manifest and manifest.is_chapter_done(chapter.index):
                cached = manifest.get_chapter_result(chapter.index)
                logger.info(f"Skipping chapter {chapter.index} (cached): {chapter.title}")
                yield AudioResult(
                    chapter=chapter,
                    output_path=cached["output_path"],
                    duration_seconds=cached["duration_seconds"],
                    generation_time_seconds=0,
                    skipped=True,
                )
            else:
                pending.append(chapter)

        if not pending:
            return

        num_workers = min(num_workers, len(pending))
        logger.info(f"Processing {len(pending)} chapters with {num_workers} parallel workers")

        # Load the model once, workers share it
        self._ensure_pipeline()

        # Thread-local storage for per-thread pipelines
        _local = threading.local()

        def get_pipeline():
            if not hasattr(_local, "pipeline"):
                _local.pipeline, _local.voice_path = self._create_worker_pipeline()
            return _local.pipeline, _local.voice_path

        def process_chapter(chapter):
            pipeline, voice_path = get_pipeline()
            return self._do_generate_chapter(
                chapter, output_dir, pipeline, voice_path,
                progress_callback, control,
            )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_chapter = {
                executor.submit(process_chapter, ch): ch for ch in pending
            }

            for future in as_completed(future_to_chapter):
                # If stopped, cancel remaining futures and break
                if control and control.stopped:
                    for f in future_to_chapter:
                        f.cancel()
                    break

                try:
                    result = future.result()
                    if manifest:
                        manifest.mark_chapter_done(
                            result.chapter.index, result.output_path, result.duration_seconds,
                        )
                    yield result
                except GenerationStopped:
                    # Worker was stopped — cancel remaining and exit
                    for f in future_to_chapter:
                        f.cancel()
                    break
                except Exception:
                    chapter = future_to_chapter[future]
                    logger.exception(f"Failed to generate chapter {chapter.index}: {chapter.title}")
                    for f in future_to_chapter:
                        f.cancel()
                    raise


def _split_long_text(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split text into segments at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    segments = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            segments.append(remaining)
            break

        chunk = remaining[:max_chars]

        split_pos = -1
        for pattern in ['. ', '? ', '! ', '.\n', '?\n', '!\n']:
            pos = chunk.rfind(pattern)
            if pos > max_chars // 4:
                split_pos = pos + len(pattern)
                break

        if split_pos == -1:
            pos = chunk.rfind('\n')
            if pos > max_chars // 4:
                split_pos = pos + 1

        if split_pos == -1:
            pos = chunk.rfind(' ')
            if pos > max_chars // 4:
                split_pos = pos + 1

        if split_pos == -1:
            split_pos = max_chars

        segments.append(remaining[:split_pos].rstrip())
        remaining = remaining[split_pos:].lstrip()

    return [s for s in segments if s.strip()]


def _safe_filename(title: str, max_len: int = 60) -> str:
    """Convert a chapter title to a safe filename."""
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe = "_".join(safe.split())
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("_")
    return safe or "untitled"
