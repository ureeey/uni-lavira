"""Unified logging configuration for LaViRA.

Three-layer control matching the architecture::

    Evaluator layer (ZS_Evaluator_mp.py):
      LAVIRA_LOG_PLAN  — branch decisions & NAV steps        (0/1, default from preset)
      LAVIRA_LOG_FMM   — FMM planner output                  (0/1)
      LAVIRA_LOG_ACT   — action execution                    (0/1)

    Agent layer (agent.py / agent_v2.py / agent_v3.py):
      LAVIRA_LOG_REQ   — API request content                 (0=off, 1=incremental, 2=full)
      LAVIRA_LOG_RESP  — prompt & response content           (0/1)

    API layer (api_openai.py / api_dashscope.py):
      LAVIRA_LOG_BODY    — HTTP body inspection              (0/1)
      LAVIRA_LOG_NETWORK — HTTP request/response metadata    (0/1)

Master preset (sets defaults for all of the above)::

    export LAVIRA_LOG=quiet    # results only
    export LAVIRA_LOG=normal   # prompts + decisions (default)
    export LAVIRA_LOG=debug    # everything

Individual overrides take precedence over the preset::

    export LAVIRA_LOG=normal
    export LAVIRA_LOG_FMM=1    # enable noisy FMM logs even in normal mode
"""

import os
import warnings

from habitat import logger

# ── Master preset ────────────────────────────────────────────────────────────
_PRESET = os.environ.get("LAVIRA_LOG", "normal").strip().lower()
if _PRESET not in ("quiet", "normal", "debug"):
    warnings.warn(f"Unknown LAVIRA_LOG={_PRESET!r}, falling back to 'normal'")
    _PRESET = "normal"

# Per-preset defaults: (quiet, normal, debug)
_PRESET_DEFAULTS = {
    "PLAN":    (0, 1, 1),
    "FMM":     (0, 0, 1),
    "ACT":     (0, 1, 1),
    "REQ":     (0, 0, 1),
    "RESP":    (0, 1, 1),
    "BODY":    (0, 0, 1),
    "NETWORK": (0, 0, 1),
}

_IDX = {"quiet": 0, "normal": 1, "debug": 2}


def _resolve(name: str, preset_default: int) -> int:
    """Read ``LAVIRA_LOG_{name}``, falling back to the preset default."""
    env_val = os.environ.get(f"LAVIRA_LOG_{name}")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            warnings.warn(f"LAVIRA_LOG_{name}={env_val!r} is not an integer, "
                          f"using preset default {preset_default}")
            return preset_default
    return preset_default


# ── Backward compatibility with old variable names ───────────────────────────
_deprecation_warnings = []  # type: list[str]

# Direct renames: old var → new flag name
_OLD_RENAMES = {
    "LAVIRA_V2_LOG_DECIDE": "PLAN",
    "LAVIRA_V2_LOG_FMM":    "FMM",
    "LAVIRA_V2_LOG_ACT":    "ACT",
    "LAVIRA_V2_LOG_REQ":    "REQ",
    "LAVIRA_V3_LOG_REQ":    "REQ",
    "LAVIRA_LOG_BODY":      "BODY",
    "LAVIRA_LOG_NETWORK":   "NETWORK",
}

for _old_var, _new_name in _OLD_RENAMES.items():
    _old_val = os.environ.get(_old_var)
    if _old_val is not None:
        _new_key = f"LAVIRA_LOG_{_new_name}"
        if os.environ.get(_new_key) is None:
            os.environ[_new_key] = _old_val
            _deprecation_warnings.append(
                f"{_old_var}={_old_val} is deprecated — "
                f"use {_new_key}={_old_val} instead"
            )

# LAVIRA_LOG_PROMPT_OUT → LOG_REQ + LOG_RESP
# Old levels: 0=full, 1=skip templates (show responses only), 2=mute all
_prompt_out = os.environ.get("LAVIRA_LOG_PROMPT_OUT")
if _prompt_out is not None:
    _level = int(_prompt_out)
    if os.environ.get("LAVIRA_LOG_REQ") is None:
        os.environ["LAVIRA_LOG_REQ"] = "1" if _level == 0 else "0"
    if os.environ.get("LAVIRA_LOG_RESP") is None:
        os.environ["LAVIRA_LOG_RESP"] = "1" if _level <= 1 else "0"
    _deprecation_warnings.append(
        f"LAVIRA_LOG_PROMPT_OUT={_level} is deprecated — "
        f"use LAVIRA_LOG_REQ + LAVIRA_LOG_RESP instead"
    )

# LAVIRA_LOG_VERBOSE — absorbed into the preset system
_old_verbose = os.environ.get("LAVIRA_LOG_VERBOSE")
if _old_verbose is not None:
    _deprecation_warnings.append(
        f"LAVIRA_LOG_VERBOSE={_old_verbose} is deprecated — "
        f"use LAVIRA_LOG=normal|quiet|debug instead"
    )

if _deprecation_warnings:
    for _w in _deprecation_warnings:
        warnings.warn(_w, DeprecationWarning, stacklevel=2)


# ── Resolve flags ────────────────────────────────────────────────────────────
_preset_idx = _IDX[_PRESET]

LOG_PLAN    = _resolve("PLAN",    _PRESET_DEFAULTS["PLAN"][_preset_idx])
LOG_FMM     = _resolve("FMM",     _PRESET_DEFAULTS["FMM"][_preset_idx])
LOG_ACT     = _resolve("ACT",     _PRESET_DEFAULTS["ACT"][_preset_idx])
LOG_REQ     = _resolve("REQ",     _PRESET_DEFAULTS["REQ"][_preset_idx])
LOG_RESP    = _resolve("RESP",    _PRESET_DEFAULTS["RESP"][_preset_idx])
LOG_BODY    = _resolve("BODY",    _PRESET_DEFAULTS["BODY"][_preset_idx])
LOG_NETWORK = _resolve("NETWORK", _PRESET_DEFAULTS["NETWORK"][_preset_idx])

# Derived: whether to use compact progress bars (quiet mode) vs text output
LOG_PROGRESS_BAR = (_PRESET == "quiet")


# ── Convenience functions ────────────────────────────────────────────────────

def log_plan(msg: str, *args):
    """Evaluator layer — branch decisions and NAV steps.  Controlled by ``LAVIRA_LOG_PLAN``."""
    if LOG_PLAN:
        if args:
            logger.info(msg, *args)
        else:
            logger.info(msg)


def log_fmm(msg: str, *args):
    """Evaluator layer — FMM planner output.  Controlled by ``LAVIRA_LOG_FMM``."""
    if LOG_FMM:
        if args:
            logger.info(msg, *args)
        else:
            logger.info(msg)


def log_act(msg: str, *args):
    """Evaluator layer — action execution.  Controlled by ``LAVIRA_LOG_ACT``."""
    if LOG_ACT:
        if args:
            logger.info(msg, *args)
        else:
            logger.info(msg)


def log_req(msg: str, *args):
    """Agent layer — API request content (prompts sent to the model).

    Controlled by ``LAVIRA_LOG_REQ``.  At level ≥1, logs the message.
    Use ``log_req_full()`` for level-2 (full conversation history) logging.
    """
    if LOG_REQ >= 1:
        if args:
            logger.info(msg, *args)
        else:
            logger.info(msg)


def log_req_full(msg: str, *args):
    """Agent layer — full request log (all messages in the conversation).

    Controlled by ``LAVIRA_LOG_REQ``.  Only logs when level ≥ 2.
    """
    if LOG_REQ >= 2:
        if args:
            logger.info(msg, *args)
        else:
            logger.info(msg)


def log_resp(msg: str, *args):
    """Agent layer — model response / output content.

    Controlled by ``LAVIRA_LOG_RESP``.
    """
    if LOG_RESP:
        if args:
            logger.info(msg, *args)
        else:
            logger.info(msg)


def log_body(messages, label=""):
    """API layer — HTTP body content structure (image count, sizes, token counts).

    Controlled by ``LAVIRA_LOG_BODY``.  Handles both OpenAI-format
    (``image_url``) and DashScope-format (``image``) message structures.
    """
    if not LOG_BODY:
        return
    img_sources = []  # type: list[str]
    total_text_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_text_chars += len(content)
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    total_text_chars += len(item["text"])
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if url.startswith("data:"):
                        img_sources.append(f"base64 ({len(url) / 1024:.0f} KB)")
                    elif url.startswith("oss://"):
                        img_sources.append(f"oss://  ({len(url):.0f} B)")
                    elif url.startswith("http"):
                        img_sources.append(f"http   ({len(url):.0f} B)")
                    else:
                        img_sources.append(f"other  ({len(url):.0f} B)")
                elif "image" in item:
                    url = item["image"]
                    if url.startswith("data:"):
                        img_sources.append(f"base64 ({len(url) / 1024:.0f} KB)")
                    elif url.startswith("oss://"):
                        img_sources.append(f"oss://  ({len(url):.0f} B)")
                    elif url.startswith("http"):
                        img_sources.append(f"http   ({len(url):.0f} B)")
                    else:
                        img_sources.append(f"other  ({len(url):.0f} B)")

    total_body_kb = total_text_chars / 1024 + sum(
        len(item.get("image_url", {}).get("url", item.get("image", ""))) / 1024
        for msg in messages
        if isinstance(msg.get("content", []), list)
        for item in msg["content"]
        if item.get("type") == "image_url" or "image" in item
    )

    summary = ", ".join(img_sources) if img_sources else "no images"
    logger.info(f"[BODY] {label}  {len(img_sources)} imgs  |  {summary}  |  "
                f"text: {total_text_chars / 1024:.0f} KB  "
                f"body: {total_body_kb:.0f} KB")


def log_network(msg: str):
    """API layer — HTTP request/response metadata (latency, status code, URL).

    Controlled by ``LAVIRA_LOG_NETWORK``.
    """
    if LOG_NETWORK:
        logger.info(f"[NET] {msg}")
