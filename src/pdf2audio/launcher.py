"""Launcher for the pdf2audio web interface.

This is the entry point when running as a standalone exe.
Starts the Flask server and opens the browser automatically.
"""

import sys
import threading
import webbrowser

from pdf2audio.web.app import app, UPLOAD_DIR, OUTPUT_DIR


def main():
    host = "127.0.0.1"
    port = 5000

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  pdf2audio - PDF to Audiobook Converter")
    print(f"  Running at http://{host}:{port}")
    print("  Close this window to stop the server.")
    print("=" * 50)
    print()

    # Open browser after a short delay to let the server start
    threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()

    # Start Flask (use_reloader=False is important for frozen exe)
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
