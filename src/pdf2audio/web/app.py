"""Flask web interface for pdf2audio."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file

from ..pdf_extract import extract_chapters, get_pdf_info
from ..tts_engine import TTSEngine, DEFAULT_VOICE, detect_device, SAMPLE_RATE
from ..audio_formats import save_audio, MP3_QUALITY_PRESETS
from ..model_manager import is_setup_complete, list_local_voices, download_models
from ..manifest import JobManifest

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit

# Global state for tracking jobs and setup
JOBS: dict[str, dict] = {}
SETUP_STATUS: dict = {"running": False, "step": 0, "total": 0, "message": "", "error": None, "done": False}
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
    quality = data.get("quality", "medium")

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
        "format": output_format,
        "quality": quality,
        "chapter_progress": {
            "chapter_index": -1,
            "chapter_title": "",
            "chars_processed": 0,
            "chars_total": 0,
        },
    }

    # Run generation in a background thread
    thread = threading.Thread(
        target=_run_generation,
        args=(job_id, chapters, str(output_dir), upload_path, voice, speed, lang_code, output_format, quality),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "running", "total_chapters": len(chapters)})


def _run_generation(
    job_id: str,
    chapters: list,
    output_dir: str,
    upload_path: str,
    voice: str,
    speed: float,
    lang_code: str,
    output_format: str,
    quality: str,
):
    """Background worker for generating audio."""
    try:
        engine = TTSEngine(
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            output_format=output_format,
            quality=quality,
        )

        # Set up manifest for resume support
        manifest = JobManifest(
            output_dir,
            pdf_path=upload_path,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            format=output_format,
            quality=quality,
        )

        # Check if manifest matches current settings, reset if not
        if manifest.manifest_path.exists() and not manifest.matches_settings(
            upload_path, voice=voice, speed=speed, lang_code=lang_code, format=output_format, quality=quality
        ):
            logger.info("Settings changed, starting fresh")
            manifest = JobManifest(output_dir, pdf_path=upload_path,
                                   voice=voice, speed=speed, lang_code=lang_code,
                                   format=output_format, quality=quality)

        def on_progress(progress):
            JOBS[job_id]["chapter_progress"] = {
                "chapter_index": progress.chapter_index,
                "chapter_title": progress.chapter_title,
                "chars_processed": progress.chars_processed,
                "chars_total": progress.chars_total,
            }

        for result in engine.generate_all(chapters, output_dir, manifest=manifest, progress_callback=on_progress):
            JOBS[job_id]["completed_chapters"] += 1
            JOBS[job_id]["results"].append({
                "chapter_index": result.chapter.index,
                "chapter_title": result.chapter.title,
                "output_path": result.output_path,
                "duration_seconds": round(result.duration_seconds, 1),
                "generation_time_seconds": round(result.generation_time_seconds, 1),
                "skipped": result.skipped,
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


@app.route("/api/download-all/<job_id>")
def download_all(job_id):
    """Concatenate all chapter audio files and serve as a single download."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Job not yet completed"}), 400

    results = sorted(job["results"], key=lambda r: r["chapter_index"])
    output_format = job.get("format", "wav")
    quality = job.get("quality", "medium")

    # Concatenate all audio with 1s silence between chapters
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second
    all_audio = []

    for r in results:
        path = r["output_path"]
        if not Path(path).exists():
            return jsonify({"error": f"Audio file missing: {path}"}), 500

        if output_format == "mp3":
            # For MP3, we need to decode to raw audio first
            # Read back using soundfile if possible, otherwise skip
            try:
                audio_data, sr = sf.read(path)
                all_audio.append(audio_data.astype(np.float32))
            except Exception:
                # MP3 files can't be read by soundfile, decode differently
                try:
                    import subprocess
                    # Use ffmpeg if available, otherwise just concatenate bytes
                    logger.warning(f"Cannot read MP3 for concatenation: {path}")
                    continue
                except Exception:
                    continue
        else:
            audio_data, sr = sf.read(path)
            all_audio.append(audio_data.astype(np.float32))

        all_audio.append(silence)

    if not all_audio:
        return jsonify({"error": "No audio data to concatenate"}), 500

    # Remove trailing silence
    if len(all_audio) > 1:
        all_audio = all_audio[:-1]

    combined = np.concatenate(all_audio)

    # Save to a temp file and serve
    combined_filename = f"audiobook_complete.{output_format}"
    combined_path = OUTPUT_DIR / job_id / combined_filename

    save_audio(combined, str(combined_path), SAMPLE_RATE, output_format, quality)

    return send_file(
        str(combined_path.resolve()),
        as_attachment=True,
        download_name=combined_filename,
    )


@app.route("/api/setup", methods=["POST"])
def start_setup():
    """Start downloading model files in the background."""
    if is_setup_complete():
        return jsonify({"status": "already_done"})

    if SETUP_STATUS["running"]:
        return jsonify({"status": "already_running"})

    SETUP_STATUS.update({"running": True, "step": 0, "total": 0, "message": "Starting...", "error": None, "done": False})

    thread = threading.Thread(target=_run_setup, daemon=True)
    thread.start()

    return jsonify({"status": "started"})


def _run_setup():
    """Background worker for downloading models."""
    try:
        def on_progress(step, total, msg):
            SETUP_STATUS.update({"step": step, "total": total, "message": msg})

        download_models(progress_callback=on_progress)
        SETUP_STATUS.update({"running": False, "done": True, "message": "Setup complete!"})
    except Exception as e:
        logger.exception("Setup failed")
        SETUP_STATUS.update({"running": False, "error": str(e), "message": f"Error: {e}"})


@app.route("/api/setup/status")
def setup_status():
    """Poll setup progress."""
    return jsonify({
        **SETUP_STATUS,
        "setup_complete": is_setup_complete(),
        "local_voices": list_local_voices() if is_setup_complete() else [],
    })


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
