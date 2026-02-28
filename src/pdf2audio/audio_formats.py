"""Audio format encoding with MP3 support via lameenc."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

# MP3 quality presets: name -> bitrate in kbps
MP3_QUALITY_PRESETS = {
    "low": 64,
    "medium": 128,
    "high": 192,
}


def save_audio(
    audio: np.ndarray,
    path: str,
    sample_rate: int = SAMPLE_RATE,
    output_format: str = "wav",
    quality: str = "medium",
) -> None:
    """Save audio data to a file in the specified format.

    For wav/flac/ogg, uses soundfile. For mp3, uses lameenc.
    """
    if output_format == "mp3":
        _save_mp3(audio, path, sample_rate, quality)
    else:
        sf.write(path, audio, sample_rate)


def _save_mp3(
    audio: np.ndarray,
    path: str,
    sample_rate: int,
    quality: str,
) -> None:
    """Encode and save audio as MP3 using lameenc."""
    try:
        import lameenc
    except ImportError:
        raise RuntimeError(
            "MP3 support requires the 'lameenc' package. "
            "Install it with: pip install lameenc"
        )

    bitrate = MP3_QUALITY_PRESETS.get(quality, 128)

    # Convert float32 [-1, 1] to int16
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)  # 2 = high quality encoding

    mp3_data = encoder.encode(audio_int16.tobytes())
    mp3_data += encoder.flush()

    Path(path).write_bytes(mp3_data)
    logger.info(f"Saved MP3: {path} ({bitrate}kbps)")
