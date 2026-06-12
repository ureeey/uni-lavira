"""Lightweight Whisper transcription server.

Run with whichever Python environment has faster_whisper installed.
Communicates via a simple HTTP API on localhost.

Usage:
    python web/whisper_server.py [PORT]

Default port: 5555
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from faster_whisper import WhisperModel

# Load model once at startup
print("[WhisperServer] Loading model...", flush=True)
model = WhisperModel("base", device="cpu", compute_type="int8")
print("[WhisperServer] Model loaded. Ready.", flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        audio_path = body.get("path", "")
        try:
            segments, _info = model.transcribe(audio_path, beam_size=5)
            text = "".join(s.text for s in segments).strip()
            resp = json.dumps({"text": text})
        except Exception as e:
            resp = json.dumps({"error": str(e)})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp.encode())

    def log_message(self, format, *args):  # noqa: A002
        pass  # suppress per-request logs


port = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
print(f"[WhisperServer] Listening on http://localhost:{port}", flush=True)
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
