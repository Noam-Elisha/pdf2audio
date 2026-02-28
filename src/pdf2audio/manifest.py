"""Job manifest for resume/checkpoint support."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class JobManifest:
    """Tracks completed chapters so jobs can be resumed after interruption.

    Thread-safe: multiple workers can call mark_chapter_done concurrently.
    """

    def __init__(self, output_dir: str, pdf_path: str | None = None, **settings):
        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "manifest.json"
        self._data: dict = {}
        self._settings = settings
        self._pdf_path = pdf_path
        self._lock = threading.Lock()

        if self.manifest_path.exists():
            self._load()
        elif pdf_path:
            self._init_new(pdf_path, settings)

    def _init_new(self, pdf_path: str, settings: dict):
        """Create a fresh manifest."""
        self._data = {
            "pdf_hash": _file_hash(pdf_path),
            "settings": settings,
            "chapters": {},
        }

    def _load(self):
        """Load existing manifest from disk."""
        try:
            self._data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            logger.info(f"Loaded manifest: {len(self._data.get('chapters', {}))} chapters recorded")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load manifest, starting fresh: {e}")
            if self._pdf_path:
                self._init_new(self._pdf_path, self._settings)
            else:
                self._data = {"pdf_hash": "", "settings": {}, "chapters": {}}

    def save(self):
        """Write manifest to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self._data, indent=2),
            encoding="utf-8",
        )

    def is_chapter_done(self, index: int) -> bool:
        """Check if a chapter has been completed and its file still exists."""
        with self._lock:
            key = str(index)
            ch = self._data.get("chapters", {}).get(key)
            if not ch:
                return False
            output_path = ch.get("output_path", "")
            if output_path and Path(output_path).exists():
                return True
            return False

    def get_chapter_result(self, index: int) -> dict | None:
        """Get stored result for a completed chapter."""
        with self._lock:
            key = str(index)
            return self._data.get("chapters", {}).get(key)

    def mark_chapter_done(self, index: int, output_path: str, duration_seconds: float):
        """Record a chapter as completed. Thread-safe."""
        with self._lock:
            if "chapters" not in self._data:
                self._data["chapters"] = {}
            self._data["chapters"][str(index)] = {
                "output_path": output_path,
                "duration_seconds": duration_seconds,
            }
            self.save()

    def matches_settings(self, pdf_path: str, **settings) -> bool:
        """Check if the manifest matches the current job settings."""
        if self._data.get("pdf_hash") != _file_hash(pdf_path):
            return False
        saved = self._data.get("settings", {})
        for key, val in settings.items():
            if str(saved.get(key)) != str(val):
                return False
        return True


def _file_hash(path: str) -> str:
    """Compute MD5 hash of a file (for change detection, not security)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
