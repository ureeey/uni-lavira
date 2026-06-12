"""
VLM Connectivity Smoke Tests for LaViRA G1
===========================================
Tests VLM endpoint connectivity using Config-sourced credentials.
No secrets are embedded; all keys and URLs are read from environment variables
via Config or the environment directly.

The test_images/ directory is not shipped with this repository.
When no test image is available, a tiny synthetic image is generated in memory
so the network / API path can still be exercised.

Usage:
    python -m pytest tests/test_vlms.py -v
    # or:
    python tests/test_vlms.py
"""

import os
import sys
import base64
import time

# Unset proxy environment variables to avoid 'socksio' error
for _key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    if _key in os.environ:
        del os.environ[_key]

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

try:
    from colorama import init, Fore, Style
    init()
    _HAS_COLORAMA = True
except ImportError:
    _HAS_COLORAMA = False
    class _FakeFore:
        CYAN = MAGENTA = GREEN = RED = ""
    class _FakeStyle:
        RESET_ALL = ""
    Fore = _FakeFore()
    Style = _FakeStyle()

from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_image_b64() -> str:
    """Generate a tiny 4x4 red PNG and return it as a base64 string.

    This allows the API path to be exercised even when no real image file is
    present on disk.
    """
    try:
        import io
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (4, 4), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        # Pillow not available; use a minimal hard-coded 1×1 red PNG (89 bytes).
        _minimal_red_png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADklEQVQI12P4"
            "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
        )
        return _minimal_red_png_b64


def _load_image_b64(img_path: str) -> str:
    """Load image from disk as base64, or synthesize one if the file is absent."""
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return _make_synthetic_image_b64()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_OPENAI, reason="openai package not installed")
def test_va_model():
    """Smoke test: VA (Vision-Action / primary) model endpoint.

    Uses Config.VA_API_KEY and Config.VA_BASE_URL.  Skipped when the endpoint
    is the default localhost (no real server expected in CI).
    """
    if Config.VA_BASE_URL.startswith("http://localhost"):
        pytest.skip("VA_BASE_URL points to localhost — skipping network test in CI")

    print(Fore.CYAN + f"\n=== Testing VA model: {Config.VA_MODEL_NAME} ===" + Style.RESET_ALL)

    client = OpenAI(api_key=Config.VA_API_KEY or "sk-no-key-required",
                    base_url=Config.VA_BASE_URL)

    # Use a real image if available, otherwise synthesize one
    img_dir = os.path.join(os.path.dirname(__file__), "test_images")
    img_path = os.path.join(img_dir, "test_rgb.png")
    img_b64 = _load_image_b64(img_path)
    source = "disk" if os.path.exists(img_path) else "synthetic"
    print(f"Image source: {source}")

    prompt = (
        'Task: Object Navigation\nInstruction: "Find the brown sofa."\n\n'
        "Output JSON with: bbox_2d [x1,y1,x2,y2] and action NAVIGATE|STOP"
    )
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        {"type": "text", "text": prompt},
    ]

    print(f"Sending request to VA endpoint ({Config.VA_BASE_URL})...")
    start = time.time()
    resp = client.chat.completions.create(
        model=Config.VA_MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        max_tokens=256,
        temperature=0.1,
    )
    duration = time.time() - start
    result = resp.choices[0].message.content
    print(Fore.GREEN + f"Success! ({duration:.2f}s)" + Style.RESET_ALL)
    print(f"Response:\n{result}")


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai package not installed")
def test_la_model():
    """Smoke test: LA (Language-Action / secondary) model endpoint.

    Uses Config.LA_API_KEY and Config.LA_BASE_URL.  Skipped when the endpoint
    is the default localhost (no real server expected in CI).
    """
    if Config.LA_BASE_URL.startswith("http://localhost"):
        pytest.skip("LA_BASE_URL points to localhost — skipping network test in CI")

    print(Fore.MAGENTA + f"\n=== Testing LA model: {Config.LA_MODEL_NAME} ===" + Style.RESET_ALL)

    client = OpenAI(api_key=Config.LA_API_KEY or "sk-no-key-required",
                    base_url=Config.LA_BASE_URL)

    # Build a minimal 4-view panorama (synthesized or from disk)
    img_dir = os.path.join(os.path.dirname(__file__), "test_images")
    panorama_dir = os.path.join(img_dir, "panorama_test")

    view_files = {
        "Front": os.path.join(panorama_dir, "view_0.png"),
        "Right": os.path.join(panorama_dir, "view_90.png"),
        "Back": os.path.join(panorama_dir, "view_180.png"),
        "Left": os.path.join(panorama_dir, "view_270.png"),
    }
    all_exist = all(os.path.exists(p) for p in view_files.values())
    source = "disk" if all_exist else "synthetic"
    print(f"Panorama image source: {source}")

    content = [{"type": "text", "text": 'Navigation Task: "Go to the kitchen."\n\nCurrent Panorama Views:'}]
    for direction, fpath in view_files.items():
        img_b64 = _load_image_b64(fpath)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
        content.append({"type": "text", "text": f"View {direction}"})

    content.append({
        "type": "text",
        "text": (
            "Analyze the views and decide which direction to go.\n"
            'Return JSON: {"turn_direction": "front"|"right"|"left"|"behind", '
            '"reasoning": "...", "stop": false}'
        ),
    })

    print(f"Sending request to LA endpoint ({Config.LA_BASE_URL})...")
    start = time.time()
    resp = client.chat.completions.create(
        model=Config.LA_MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        temperature=0.1,
    )
    duration = time.time() - start
    result = resp.choices[0].message.content
    print(Fore.GREEN + f"Success! ({duration:.2f}s)" + Style.RESET_ALL)
    print(f"Response:\n{result}")


if __name__ == "__main__":
    print("Starting VLM Connection Tests...")
    test_va_model()
    test_la_model()
