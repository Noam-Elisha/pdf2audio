# pdf2audio

Convert PDFs to audiobooks with chapter-by-chapter audio output using [Kokoro TTS](https://github.com/hexgrad/kokoro).

- Automatically detects chapters from PDF table of contents or headings
- Outputs each chapter as a separate audio file as it completes
- Start listening to Chapter 1 while later chapters are still generating
- Works with GPU (CUDA, MPS) or CPU — auto-detects the fastest option
- Both CLI and web interface

## Installation

**Requirements:** Python 3.10-3.12, [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases)

```bash
# Install espeak-ng (required by Kokoro)
# Windows: download .msi from https://github.com/espeak-ng/espeak-ng/releases
# Linux: apt-get install espeak-ng
# macOS: brew install espeak-ng

# Clone and install
git clone https://github.com/YOUR_USERNAME/pdf2audio.git
cd pdf2audio
pip install -e .
```

For GPU acceleration, install PyTorch with CUDA support first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## CLI Usage

### Convert a PDF to audio

```bash
pdf2audio convert book.pdf
```

Audio files appear in `./output/book/` as they're generated. Start playing `000_Chapter_1.wav` immediately.

### Options

```bash
pdf2audio convert book.pdf \
  --voice af_heart \
  --speed 1.0 \
  --lang a \
  --format wav \
  --output ./my-audiobook/ \
  --device cuda
```

### Inspect a PDF (see chapters without generating)

```bash
pdf2audio inspect book.pdf
```

### List voices

```bash
pdf2audio voices
```

### Check compute devices

```bash
pdf2audio devices
```

## Web Interface

```bash
pdf2audio web
# Opens at http://127.0.0.1:5000
```

The web UI lets you:
1. Upload a PDF
2. Configure voice, speed, language, and format
3. See detected chapters
4. Generate audio and play chapters as they complete

## How It Works

1. **PDF Parsing** — Extracts chapters using PDF table of contents (bookmarks). Falls back to heuristic heading detection if no TOC exists.
2. **Text Extraction** — Pulls clean text from each chapter's page range using PyMuPDF.
3. **TTS Generation** — Feeds chapter text to Kokoro TTS, which generates 24kHz audio using a generator pattern.
4. **Streaming Output** — Each chapter's audio is saved to disk as soon as generation completes, so playback can begin immediately.

## Voices

| Voice | Language | Gender | Quality |
|-------|----------|--------|---------|
| `af_heart` | American English | Female | A |
| `af_bella` | American English | Female | A- |
| `am_adam` | American English | Male | B |
| `bf_emma` | British English | Female | B- |
| `ff_siwis` | French | Female | B- |

Run `pdf2audio voices` for the full list.

## License

MIT
