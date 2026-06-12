"""
Flask + SocketIO interactive demo for the Unitree Go1 VLN task.

Users either type an instruction or press-and-hold the microphone button to
dictate one.  The single VLN controller runs the navigation loop on a
background thread; status / response messages are pushed back via SocketIO.

Whisper transcription uses a 3-tier fallback (identical to source
web_interface.py):
  1. In-process faster-whisper base int8 model (set via set_whisper_model).
  2. Subprocess whisper_server.py HTTP API on port 5555.
  3. Remote OpenAI whisper-1 API (key from env WHISPER_API_KEY).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.request

try:
    import cv2  # type: ignore[import]
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import simplejpeg  # type: ignore[import]
except ImportError:
    simplejpeg = None

from flask import Flask, Response, render_template
from flask_socketio import SocketIO

from config import Config

# ---------------------------------------------------------------------------
# Module-level app / socketio — importable without a running server
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = Config.FLASK_SECRET_KEY

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=60,
    ping_interval=25,
)

# ---------------------------------------------------------------------------
# Injected by main.py after hardware init
# ---------------------------------------------------------------------------
controller = None          # IntegratedVisionNavController
_whisper_model = None      # faster-whisper WhisperModel instance (tier-1)

_WHISPER_PORT = 5555

# Tier-3 remote Whisper credentials — sourced from environment only
_WHISPER_API_KEY  = os.environ.get("WHISPER_API_KEY", "")
_WHISPER_BASE_URL = os.environ.get("WHISPER_BASE_URL", "https://api.openai.com/v1")

# Conda / custom python for launching whisper_server subprocess
_WHISPER_PYTHON = os.environ.get("WHISPER_PYTHON", sys.executable)


def set_controller(ctrl) -> None:
    """Inject the navigation controller (called by main.py)."""
    global controller
    controller = ctrl


def set_whisper_model(model) -> None:
    """Inject a pre-loaded faster-whisper model (called by main.py)."""
    global _whisper_model
    _whisper_model = model


def run_server() -> None:
    """Start the Flask-SocketIO server.

    SSL auto-detect: uses Config.SSL_CERT_PATH / SSL_KEY_PATH when both files
    exist (mirrors source web_interface.run()).  Falls back to plain HTTP when
    the certificate files are absent.

    Called by main.py after set_controller() / set_whisper_model().
    """
    host = Config.SERVER_HOST
    port = Config.SERVER_PORT
    cert = Config.SSL_CERT_PATH
    key  = Config.SSL_KEY_PATH

    ssl_context = None
    if os.path.exists(cert) and os.path.exists(key):
        ssl_context = (cert, key)
        protocol = "https"
    else:
        protocol = "http"
        print(
            "[Web] SSL certificates not found — running in HTTP mode "
            "(voice input may not work in some browsers)."
        )

    print(f"\n{'='*60}")
    print(f"Starting web interface on {protocol}://{host}:{port}")
    print(f"{'='*60}\n")

    kwargs = {"host": host, "port": port, "debug": False, "use_reloader": False}
    if ssl_context:
        kwargs["ssl_context"] = ssl_context

    socketio.run(app, **kwargs)


# ---------------------------------------------------------------------------
# Pages and video feed
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Stream the current front camera as MJPEG."""

    def generate():
        while True:
            frame = None
            if controller is not None:
                try:
                    frame = controller.get_front_image_jpeg()
                except Exception:
                    frame = None

            if frame is not None:
                # controller.get_front_image_jpeg() may return raw JPEG bytes
                # or a numpy BGR frame depending on the robot backend.
                if isinstance(frame, (bytes, bytearray)):
                    jpeg_bytes = bytes(frame)
                else:
                    # numpy array — encode here
                    jpeg_bytes = _encode_frame(frame)

                if jpeg_bytes:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg_bytes
                        + b"\r\n"
                    )

            time.sleep(0.05)  # ~20 FPS max

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _encode_frame(frame) -> bytes | None:
    """Encode a numpy BGR frame to JPEG bytes (simplejpeg preferred, cv2 fallback)."""
    if simplejpeg is not None:
        try:
            return simplejpeg.encode_jpeg(frame, quality=70, colorspace="BGR", fastdct=True)
        except Exception:
            pass
    if cv2 is not None:
        try:
            ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                return buf.tobytes()
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# WebSocket handlers
# ---------------------------------------------------------------------------
@socketio.on("connect")
def handle_connect():
    print("[Web] Client connected")
    socketio.emit("response", {"message": "Connected to LaViRA Robot. Ready for commands."})


@socketio.on("disconnect")
def handle_disconnect():
    print("[Web] Client disconnected")


@socketio.on("text_command")
def handle_text_command(data):
    """Run VLN from a typed instruction. Payload: ``{'instruction': '<text>'}``."""
    instruction = (data or {}).get("instruction", "").strip()
    if not instruction:
        socketio.emit("response", {"message": "Please type an instruction first."})
        return
    if controller is None:
        socketio.emit("response", {"message": "Robot not initialised yet."})
        return
    controller.start_new_task(instruction)


@socketio.on("audio_command")
def handle_audio_command(audio_bytes):
    """Transcribe the recorded audio via 3-tier Whisper fallback, then run VLN."""
    process_audio(audio_bytes)


# ---------------------------------------------------------------------------
# Audio processing — 3-tier Whisper fallback (mirrors source web_interface.py)
# ---------------------------------------------------------------------------
def _probe_whisper_server() -> bool:
    """Return True if a whisper_server.py is reachable on _WHISPER_PORT."""
    try:
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
        req = urllib.request.Request(
            f"http://127.0.0.1:{_WHISPER_PORT}",
            data=b'{"path":""}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener.open(req, timeout=2)
        return True
    except Exception:
        return False


def _transcribe_subprocess(audio_path: str) -> str | None:
    """Tier-2: call the whisper_server.py HTTP API (bypasses proxy)."""
    try:
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
        abs_path = os.path.abspath(audio_path)
        data = json.dumps({"path": abs_path}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{_WHISPER_PORT}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        if "error" in result:
            print(f"[Whisper] Subprocess server error: {result['error']}")
            return None
        return result.get("text", "")
    except Exception as e:
        print(f"[Whisper] Subprocess server request failed: {e}")
        return None


def _transcribe_remote(audio_path: str) -> str | None:
    """Tier-3: use OpenAI whisper-1 API as last-resort fallback."""
    if not _WHISPER_API_KEY:
        print("[Whisper] WHISPER_API_KEY not set; skipping remote fallback.")
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=_WHISPER_API_KEY, base_url=_WHISPER_BASE_URL)
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return transcription.text
    except Exception as e:
        print(f"[Whisper] Remote transcription failed: {e}")
        return None


def process_audio(audio_data: bytes) -> None:
    """
    3-tier transcription then VLN dispatch.

    Tier 1 — in-process faster-whisper (set via set_whisper_model).
    Tier 2 — subprocess whisper_server.py HTTP on port 5555.
    Tier 3 — remote OpenAI whisper-1 (key from env WHISPER_API_KEY).
    """
    print("[Web] Received audio data...")

    tmp_fd, temp_filename = tempfile.mkstemp(suffix=".webm")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(audio_data)

        socketio.emit("status_update", {"message": "Processing voice command..."})

        text = None

        # Tier 1: in-process faster-whisper
        if _whisper_model is not None:
            try:
                segments, _info = _whisper_model.transcribe(temp_filename, beam_size=5)
                text = "".join(seg.text for seg in segments).strip()
                print(f"[Whisper] Tier-1 (in-process): '{text}'")
            except Exception as e:
                print(f"[Whisper] Tier-1 failed: {e}")

        # Tier 2: subprocess whisper_server.py
        if not text:
            print("[Whisper] Attempting tier-2 (subprocess server)...")
            server_up = _probe_whisper_server()
            if server_up:
                text = _transcribe_subprocess(temp_filename)
                if text:
                    print(f"[Whisper] Tier-2 (subprocess): '{text}'")
            else:
                print(
                    f"[Whisper] No server on port {_WHISPER_PORT}. "
                    f"Start with: {_WHISPER_PYTHON} web/whisper_server.py {_WHISPER_PORT}"
                )

        # Tier 3: remote OpenAI API
        if not text:
            print("[Whisper] Attempting tier-3 (remote API)...")
            text = _transcribe_remote(temp_filename)
            if text:
                print(f"[Whisper] Tier-3 (remote): '{text}'")

        print(f"[Web] Transcribed: {text!r}")

        if text:
            socketio.emit("status_update", {"message": f"Heard: '{text}'"})
            if controller is not None:
                controller.start_new_task(text)
            else:
                socketio.emit("response", {"message": "Robot not initialised yet."})
        else:
            socketio.emit(
                "response",
                {"message": "Could not understand audio (voice recognition unavailable)."},
            )

    except Exception as e:
        print(f"[Web] Error processing audio: {e}")
        socketio.emit("response", {"message": f"Error: {e}"})
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
