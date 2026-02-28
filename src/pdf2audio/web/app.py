"""Flask web interface for pdf2audio."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory

from ..pdf_extract import extract_chapters, get_pdf_info
from ..tts_engine import TTSEngine, DEFAULT_VOICE, detect_device
from ..model_manager import is_setup_complete, list_local_voices

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit

# Global state for tracking jobs
JOBS: dict[str, dict] = {}
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")


@app.route("/")
def index():
    device = detect_device()
    setup_done = is_setup_complete()
    local_voices = list_local_voices() if setup_done else []
    return render_template(
        "index.html",
        device=device,
        default_voice=DEFAULT_VOICE,
        setup_done=setup_done,
        local_voices=local_voices,
    )


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    """Upload a PDF and extract its chapters."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Save with unique name
    job_id = str(uuid.uuid4())[:8]
    safe_name = "".join(c if c.isalnum() or c in ".-_ " else "_" for c in file.filename)
    upload_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    file.save(str(upload_path))

    # Extract chapters
    try:
        info = get_pdf_info(str(upload_path))
        chapters = extract_chapters(str(upload_path))
    except Exception as e:
        upload_path.unlink(missing_ok=True)
        return jsonify({"error": f"Failed to process PDF: {e}"}), 400

    chapter_data = [
        {
            "index": ch.index,
            "title": ch.title,
            "page_start": ch.page_start,
            "page_end": ch.page_end,
            "char_count": len(ch.text),
            "est_minutes": round(len(ch.text) / (150 * 5), 1),
        }
        for ch in chapters
    ]

    return jsonify({
        "job_id": job_id,
        "filename": file.filename,
        "pages": info["pages"],
        "chapters": chapter_data,
        "upload_path": str(upload_path),
    })


@app.route("/api/generate", methods=["POST"])
def start_generation():
    """Start audio generation for a previously uploaded PDF."""
    data = request.json
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    job_id = data.get("job_id")
    upload_path = data.get("upload_path")
    voice = data.get("voice", DEFAULT_VOICE)
    speed = float(data.get("speed", 1.0))
    lang_code = data.get("lang_code", "a")
    output_format = data.get("format", "wav")

    if not upload_path or not os.path.exists(upload_path):
        return jsonify({"error": "Upload not found"}), 404

    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize job status
    chapters = extract_chapters(upload_path)
    JOBS[job_id] = {
        "status": "running",
        "total_chapters": len(chapters),
        "completed_chapters": 0,
        "results": [],
        "error": None,
        "started_at": time.time(),
    }

    # Run generation in a background thread
    thread = threading.Thread(
        target=_run_generation,
        args=(job_id, chapters, str(output_dir), voice, speed, lang_code, output_format),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "running", "total_chapters": len(chapters)})


def _run_generation(
    job_id: str,
    chapters: list,
    output_dir: str,
    voice: str,
    speed: float,
    lang_code: str,
    output_format: str,
):
    """Background worker for generating audio."""
    try:
        engine = TTSEngine(
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            output_format=output_format,
        )

        for result in engine.generate_all(chapters, output_dir):
            JOBS[job_id]["completed_chapters"] += 1
            JOBS[job_id]["results"].append({
                "chapter_index": result.chapter.index,
                "chapter_title": result.chapter.title,
                "output_path": result.output_path,
                "duration_seconds": round(result.duration_seconds, 1),
                "generation_time_seconds": round(result.generation_time_seconds, 1),
            })

        JOBS[job_id]["status"] = "completed"

    except Exception as e:
        logger.exception(f"Generation failed for job {job_id}")
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


@app.route("/api/status/<job_id>")
def job_status(job_id):
    """Poll for job status. Returns completed chapters as they finish."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/audio/<job_id>/<filename>")
def serve_audio(job_id, filename):
    """Serve a generated audio file."""
    audio_dir = OUTPUT_DIR / job_id
    if not audio_dir.exists():
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(str(audio_dir.resolve()), filename)


@app.route("/api/device")
def device_info():
    """Return detected compute device and setup status."""
    return jsonify({
        "device": detect_device(),
        "setup_complete": is_setup_complete(),
        "local_voices": list_local_voices(),
    })


def create_app(host="127.0.0.1", port=5000, debug=False):
    """Create and run the Flask app."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host=host, port=port, debug=debug)
