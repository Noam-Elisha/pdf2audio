"""Manage local model files for offline Kokoro TTS usage."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Default local model directory
DEFAULT_MODEL_DIR = Path.home() / ".pdf2audio" / "models"

# Files needed from the HuggingFace repo
REPO_ID = "hexgrad/Kokoro-82M"
CONFIG_FILE = "config.json"
MODEL_FILE = "kokoro-v1_0.pth"

# Voices to download by default (highest quality ones)
DEFAULT_VOICES = [
    "af_heart", "af_bella", "af_sarah", "af_nicole", "af_aoede", "af_kore", "af_sky",
    "am_adam", "am_michael", "am_fenrir",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
    "ff_siwis",
    "jf_alpha",
]


def get_model_dir() -> Path:
    """Get the model directory, respecting PDF2AUDIO_MODEL_DIR env var."""
    env_dir = os.environ.get("PDF2AUDIO_MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    return DEFAULT_MODEL_DIR


def is_setup_complete(model_dir: Path | None = None) -> bool:
    """Check if model files have been downloaded."""
    model_dir = model_dir or get_model_dir()
    config_path = model_dir / CONFIG_FILE
    model_path = model_dir / MODEL_FILE
    voices_dir = model_dir / "voices"
    return (
        config_path.exists()
        and model_path.exists()
        and voices_dir.exists()
        and any(voices_dir.glob("*.pt"))
    )


def get_local_paths(model_dir: Path | None = None) -> dict:
    """Get paths to local model files."""
    model_dir = model_dir or get_model_dir()
    return {
        "config": str(model_dir / CONFIG_FILE),
        "model": str(model_dir / MODEL_FILE),
        "voices_dir": str(model_dir / "voices"),
    }


def download_models(
    model_dir: Path | None = None,
    voices: list[str] | None = None,
    progress_callback=None,
) -> Path:
    """Download model files from HuggingFace to a local directory.

    This is a one-time setup step. After this, everything runs locally.
    """
    from huggingface_hub import hf_hub_download

    model_dir = model_dir or get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    voices_dir = model_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    voices = voices or DEFAULT_VOICES
    total_steps = 2 + len(voices)  # config + model + voices
    step = 0

    def _report(msg: str):
        nonlocal step
        step += 1
        logger.info(f"[{step}/{total_steps}] {msg}")
        if progress_callback:
            progress_callback(step, total_steps, msg)

    # Download config.json
    _report("Downloading config.json...")
    config_src = hf_hub_download(repo_id=REPO_ID, filename=CONFIG_FILE)
    shutil.copy2(config_src, model_dir / CONFIG_FILE)

    # Download model weights
    _report(f"Downloading {MODEL_FILE} (~330 MB)...")
    model_src = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
    shutil.copy2(model_src, model_dir / MODEL_FILE)

    # Download voice files
    for voice_name in voices:
        _report(f"Downloading voice: {voice_name}")
        voice_src = hf_hub_download(repo_id=REPO_ID, filename=f"voices/{voice_name}.pt")
        shutil.copy2(voice_src, voices_dir / f"{voice_name}.pt")

    logger.info(f"Setup complete. Models saved to {model_dir}")
    return model_dir


def get_voice_path(voice_name: str, model_dir: Path | None = None) -> str:
    """Get the local path for a voice file (.pt)."""
    model_dir = model_dir or get_model_dir()
    voice_path = model_dir / "voices" / f"{voice_name}.pt"
    if not voice_path.exists():
        raise FileNotFoundError(
            f"Voice '{voice_name}' not found at {voice_path}. "
            f"Run 'pdf2audio setup' to download models."
        )
    return str(voice_path)


def list_local_voices(model_dir: Path | None = None) -> list[str]:
    """List voice names available locally."""
    model_dir = model_dir or get_model_dir()
    voices_dir = model_dir / "voices"
    if not voices_dir.exists():
        return []
    return sorted(p.stem for p in voices_dir.glob("*.pt"))
