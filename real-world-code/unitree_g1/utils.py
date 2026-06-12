"""
LaViRA Utility Functions for Unitree G1
========================================
Shared utilities for logging, image processing, and JSON parsing.
"""

import os
import json
import base64
import re
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image as PILImage

try:
    from colorama import Fore, Style, init as _colorama_init
    _colorama_init(autoreset=True)
except ImportError:  # pragma: no cover - optional dependency
    class _NoColour:
        RESET = CYAN = GREEN = BLUE = YELLOW = RED = MAGENTA = WHITE = ""

        def __getattr__(self, _name: str) -> str:
            return ""

    Fore = _NoColour()
    Style = _NoColour()


# =========================================================================
# Logging Utilities
# =========================================================================

def print_step(step_num: int, description: str) -> None:
    """Print step information."""
    print(Fore.CYAN + f"\n{'='*60}")
    print(Fore.CYAN + f"[STEP {step_num}] {description}")
    print(Fore.CYAN + f"{'='*60}")


def print_action(action: str, details: str = "") -> None:
    """Print action information."""
    print(Fore.GREEN + f"  [ACTION] {action}" + (f" - {details}" if details else ""))


def print_info(info: str) -> None:
    """Print general information."""
    print(Fore.BLUE + f"  [INFO] {info}")


def print_warning(warning: str) -> None:
    """Print warning information."""
    print(Fore.YELLOW + f"  [WARNING] {warning}")


def print_error(error: str) -> None:
    """Print error information."""
    print(Fore.RED + f"  [ERROR] {error}")


def print_success(success: str) -> None:
    """Print success information."""
    print(Fore.GREEN + f"  [SUCCESS] {success}")


def print_model_response(response_type: str, content: str) -> None:
    """Print model response information."""
    print(Fore.MAGENTA + f"  [MODEL {response_type}] {content}")


def print_robot(message: str) -> None:
    """Print G1 robot-specific status."""
    print(Fore.WHITE + f"  [G1 ROBOT] {message}")


def print_model_interaction(
    model_name: str,
    prompt: str,
    response: str,
    speed: Optional[float] = None,
    duration: Optional[float] = None,
    prompt_speed: Optional[float] = None,
) -> None:
    """Pretty-print a model call.

    Truncates base64 image payloads in the prompt for readability.

    Args:
        model_name: Name of the model being called.
        prompt: The full prompt sent to the model.
        response: The model's response text.
        speed: Output token generation speed (tokens/s), if available.
        duration: Total call duration in seconds, if available.
        prompt_speed: Input token processing speed (tokens/s), if available.
    """
    print(Fore.YELLOW + "=" * 60)
    print(Fore.YELLOW + f"MODEL: {model_name}")
    print(Fore.CYAN + "-" * 20 + " PROMPT " + "-" * 20)

    clean_prompt = str(prompt)
    if "data:image" in clean_prompt:
        clean_prompt = re.sub(
            r"data:image/[^;]+;base64,[a-zA-Z0-9+/=]+",
            "[IMAGE_BASE64_DATA]",
            clean_prompt,
        )
    print(clean_prompt)

    print(Fore.GREEN + "-" * 20 + " RESPONSE " + "-" * 20)
    print(response)

    if any(v is not None for v in (speed, duration, prompt_speed)):
        print(Fore.MAGENTA + "-" * 20 + " STATS " + "-" * 20)
        if prompt_speed is not None:
            print(Fore.MAGENTA + f"Input speed:  {prompt_speed:.2f} tokens/s")
        if speed is not None:
            print(Fore.MAGENTA + f"Output speed: {speed:.2f} tokens/s")
        if duration is not None:
            print(Fore.MAGENTA + f"Duration:     {duration:.2f}s")
    print(Fore.YELLOW + "=" * 60)


# =========================================================================
# File I/O Utilities
# =========================================================================

def save_output(output_dir: str, filename: str, content: Union[Dict, List, str]) -> None:
    """Save output to a local file.

    Args:
        output_dir: Directory path where the file will be written.
        filename: Name of the output file.
        content: Data to write; dicts and lists are serialised as JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if isinstance(content, (dict, list)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(content))
    print_action(f"Saved output file: {filename}")


# =========================================================================
# Image Utilities
# =========================================================================

def numpy_to_base64(img_np: np.ndarray) -> str:
    """Convert a NumPy array (OpenCV BGR image) to a base64 JPEG string.

    Args:
        img_np: BGR image as a NumPy array.

    Returns:
        Base64-encoded JPEG string, or empty string on failure.
    """
    if img_np is None:
        return ""
    _, buffer = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text


def img_to_base64(img_path: str) -> str:
    """Read an image file and convert it to a base64 PNG string.

    Args:
        img_path: Absolute or relative path to the image file.

    Returns:
        Base64-encoded PNG string, or empty string on failure.
    """
    if not os.path.exists(img_path):
        print_error(f"Image file not found: {img_path}")
        return ""
    try:
        img = PILImage.open(img_path).convert("RGB")
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print_error(f"Failed to convert image to base64: {e}")
        return ""


# =========================================================================
# JSON Parsing Utilities
# =========================================================================

def safe_json_loads(text: str) -> Dict[str, Any]:
    """Safely parse LLM output as JSON, automatically fixing common format errors.

    Falls back through multiple layers:
    1. Direct ``json.loads`` parse.
    2. Extract JSON from a markdown code block.
    3. Extract the first ``{...}`` block and fix common syntax errors.
    4. Manual regex extraction of known key fields.

    Args:
        text: Raw text output from a language model.

    Returns:
        Parsed dictionary, or empty dict if all layers fail.
    """
    # Layer 1: Try direct parse
    try:
        result = json.loads(text)
        return result
    except json.JSONDecodeError:
        pass

    # Layer 2: Try extracting JSON from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        try:
            result = json.loads(code_block_match.group(1).strip())
            return result
        except json.JSONDecodeError:
            pass

    # Layer 3: Extract JSON object (handle nested braces)
    brace_count = 0
    start_idx = None
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                json_str = text[start_idx:i + 1]
                try:
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    break

    # Layer 4: Try simple regex extraction
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not json_match:
        print_error("JSON parsing failed - No JSON structure found")
        print_warning(f"[Original Content]:\n{text[:500]}")
        return {}

    json_str = json_match.group()

    # Layer 5: Fix common JSON errors
    try:
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r'(\{|\,\s*)(\w+)\s*:', r'\1"\2":', json_str)
        json_str = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)(\s*[,}])', r':"\1"\2', json_str)
        json_str = re.sub(r':\s*(true|false|null)\s*([,}])', r':\1\2', json_str)
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        json_str = re.sub(r':\s*(\d+\.?\d*)\s*([,}])', r':\1\2', json_str)

        result = json.loads(json_str)
        print_success("JSON parsed successfully (after fix)")
        return result

    except json.JSONDecodeError as e:
        print_error(f"JSON parsing failed: {e}")

        # Layer 6: Fallback — manual extraction of known key fields
        try:
            result = {}
            for key in ["turn_direction", "reasoning", "description", "action",
                        "stop", "bbox_2d", "visual_check", "target"]:
                match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
                if match:
                    result[key] = match.group(1).strip()

            # Special handling for bbox_2d array
            bbox_match = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', text)
            if bbox_match:
                try:
                    coords = [int(x.strip()) for x in bbox_match.group(1).split(',')]
                    result["bbox_2d"] = coords
                except ValueError:
                    pass

            # Special handling for stop boolean
            stop_match = re.search(r'"stop"\s*:\s*(true|false)', text, re.IGNORECASE)
            if stop_match:
                result["stop"] = stop_match.group(1).lower() == "true"

            if result:
                print_success("Manual extraction of key info successful")
                return result
            else:
                return {}

        except Exception as e2:
            print_error(f"Manual extraction also failed: {e2}")
            return {}
