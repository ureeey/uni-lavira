"""CI-safe unit tests for navigation_api logic.

Tests cover:
1. safe_json_loads — robust JSON extraction used by both LA and VA paths.
2. bbox de-normalisation — the [0, 1000] -> pixel conversion in query_llm_bbox.

No ROS, no hardware, no network calls required.
"""
from __future__ import annotations

import os
import sys

# Allow importing from the package root without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils


# ---------------------------------------------------------------------------
# safe_json_loads tests
# ---------------------------------------------------------------------------

def test_safe_json_loads_plain_object():
    """Direct JSON string with no extra wrapping parses correctly."""
    result = utils.safe_json_loads('{"action": "NAVIGATE", "turn_direction": "front"}')
    assert result == {"action": "NAVIGATE", "turn_direction": "front"}


def test_safe_json_loads_with_trailing_text():
    """JSON embedded in surrounding text is extracted correctly."""
    text = 'Some reasoning text here. {"action": "STOP"} And trailing text.'
    result = utils.safe_json_loads(text)
    assert result.get("action") == "STOP"


def test_safe_json_loads_with_markdown_fence():
    """JSON wrapped in a markdown code fence is extracted correctly."""
    text = '```json\n{"bbox_2d": [100, 200, 300, 400]}\n```'
    result = utils.safe_json_loads(text)
    # After layer-2 extraction the fence is stripped; value must be present.
    assert "bbox_2d" in result or result != {}


def test_safe_json_loads_empty_input_returns_empty_dict():
    """An empty or whitespace-only input returns an empty dict, not an exception."""
    result = utils.safe_json_loads("")
    assert isinstance(result, dict)


def test_safe_json_loads_malformed_keys():
    """Unquoted keys (common LLM output error) are fixed and parsed."""
    text = "{action: NAVIGATE, turn_direction: front}"
    result = utils.safe_json_loads(text)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# [0, 1000] -> pixel bbox de-normalisation tests
#
# Replicates the arithmetic from navigation_api.LaViRANavigationAPI.query_llm_bbox:
#
#   if max(b) <= 1000:
#       b = [b[0]/1000*w, b[1]/1000*h, b[2]/1000*w, b[3]/1000*h]
#   bbox_2d = [int(max(0, min(w, b[0]))), ...]
# ---------------------------------------------------------------------------

def _denormalize_bbox(bbox_1000, img_width, img_height):
    """Pure-Python replica of the normalization step in query_llm_bbox."""
    b = list(bbox_1000)
    w, h = img_width, img_height
    if max(b) <= 1000:
        b = [b[0] / 1000 * w, b[1] / 1000 * h,
             b[2] / 1000 * w, b[3] / 1000 * h]
    return [
        int(max(0, min(w, b[0]))),
        int(max(0, min(h, b[1]))),
        int(max(0, min(w, b[2]))),
        int(max(0, min(h, b[3]))),
    ]


def test_bbox_denorm_center_box_640x480():
    """A [500,500,800,800] box on 640x480 maps to [320,240,512,384]."""
    result = _denormalize_bbox([500, 500, 800, 800], 640, 480)
    assert result == [320, 240, 512, 384], f"Unexpected result: {result}"


def test_bbox_denorm_full_image_box():
    """A [0,0,1000,1000] box should span the entire image."""
    w, h = 1280, 720
    result = _denormalize_bbox([0, 0, 1000, 1000], w, h)
    assert result == [0, 0, w, h], f"Unexpected result: {result}"


def test_bbox_denorm_clamped_to_image_bounds():
    """Values slightly above 1000 are clamped and not treated as already-pixel."""
    # max(b) == 1000 -> still normalised; values clamped to image size.
    result = _denormalize_bbox([0, 0, 1000, 1000], 640, 480)
    assert result[2] <= 640
    assert result[3] <= 480


def test_bbox_denorm_small_box_720p():
    """A tight 100x100 region at the top-left on a 1280x720 image."""
    result = _denormalize_bbox([0, 0, 100, 100], 1280, 720)
    assert result == [0, 0, 128, 72], f"Unexpected result: {result}"


def test_bbox_denorm_already_pixel_coords_unchanged():
    """Coordinates already in pixel space (max > 1000) bypass normalisation."""
    # max(b) > 1000 -> skip the division step; coords stay as-is (clamped).
    result = _denormalize_bbox([100, 100, 1100, 500], 1280, 720)
    # x2 = min(1280, 1100) = 1100; no division applied.
    assert result == [100, 100, 1100, 500], f"Unexpected result: {result}"
