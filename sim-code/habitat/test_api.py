#!/usr/bin/env python
"""Quick API latency test — minimal request, detailed timing.

Usage:
    source .env.api
    python test_api.py --model qwen3.7-plus
    python test_api.py --model qwen3.7-max
    python test_api.py --model deepseek-v4-pro
    python test_api.py --model deepseek-v4-flash
    python test_api.py --model qwen3.7-plus --repeat 5
"""

import os, sys, time, argparse, base64, struct, zlib, json, threading, socket, ssl
from typing import Optional, List, Dict, Tuple
from urllib.parse import urlparse

import httpx
from openai import OpenAI

# ── minimal test payload ──────────────────────────────────────────────────

MINIMAL_TEXT = "Say exactly: OK"

def _make_test_image(width: int = 20, height: int = 20,
                     r: int = 128, g: int = 128, b: int = 128) -> str:
    """Generate a minimal valid PNG as a base64 data-URI."""
    # Build raw pixel data (filter byte + RGB per row)
    raw = b''
    for _ in range(height):
        raw += b'\x00'  # filter: none
        for _ in range(width):
            raw += bytes([r, g, b])

    compressed = zlib.compress(raw)

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        return struct.pack('>I', len(data)) + c + crc

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = _chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    idat = _chunk(b'IDAT', compressed)
    iend = _chunk(b'IEND', b'')

    b64 = base64.b64encode(sig + ihdr + idat + iend).decode()
    return f"data:image/png;base64,{b64}"

MINIMAL_IMAGE = _make_test_image()

# ── model catalog ────────────────────────────────────────────────────────────
# Built-in fallback list.  Use --models-file PATH to maintain your own
# up-to-date catalog (run with --fetch first to populate it).
#
# Usage:
#   python test_api.py --list-models --fetch --models-file ~/.lavira_models.json
#   python test_api.py --list-models --models-file ~/.lavira_models.json

_BUILTIN_MODELS: Dict[str, List[Dict[str, str]]] = {
    "QWEN": [
        {"name": "qwen3.7-max",           "type": "text"},
        {"name": "qwen3.7-plus",          "type": "vision"},
        {"name": "qwen3.7-max-preview",   "type": "text"},
        {"name": "qwen3.6-plus",          "type": "vision"},
        {"name": "qwen3.6-flash",         "type": "vision"},
        {"name": "qwen3.6-max-preview",   "type": "text"},
        {"name": "qwen3.5-plus",          "type": "vision"},
        {"name": "qwen3.5-flash",         "type": "vision"},
        {"name": "qwen3.5-ocr",           "type": "vision"},
        {"name": "qwen3.5-omni-flash",    "type": "vision"},
        {"name": "qwen3.5-omni-plus",     "type": "vision"},
        {"name": "qwen3-max",             "type": "text"},
        {"name": "qwen3-coder-plus",      "type": "text"},
        {"name": "qwen3-coder-flash",     "type": "text"},
        {"name": "qwen3-coder-next",      "type": "text"},
        {"name": "qwen3-vl-flash",        "type": "vision"},
        {"name": "qwen3-vl-plus",         "type": "vision"},
        {"name": "qwen3-omni-flash",      "type": "vision"},
        {"name": "qwen-max",              "type": "text"},
        {"name": "qwen-plus",             "type": "vision"},
        {"name": "qwen-turbo",            "type": "text"},
        {"name": "qwen-long",             "type": "text"},
        {"name": "qwen-flash",            "type": "vision"},
        {"name": "qwen-coder-plus",       "type": "text"},
        {"name": "qwen-coder-turbo",      "type": "text"},
        {"name": "qwen-vl-max",           "type": "vision"},
        {"name": "qwen-vl-plus",          "type": "vision"},
        {"name": "qwen-vl-ocr",           "type": "vision"},
        {"name": "qwen-omni-turbo",       "type": "vision"},
        {"name": "qvq-max",               "type": "vision"},
        {"name": "qvq-plus",              "type": "vision"},
        {"name": "qwq-plus",              "type": "vision"},
        {"name": "qwen-math-plus",        "type": "vision"},
        {"name": "qwen-math-turbo",       "type": "text"},
        {"name": "deepseek-v4-pro",       "type": "text"},
        {"name": "deepseek-v4-flash",     "type": "text"},
        {"name": "deepseek-v3",           "type": "text"},
        {"name": "deepseek-v3.1",         "type": "text"},
        {"name": "deepseek-v3.2",         "type": "text"},
        {"name": "deepseek-r1",           "type": "text"},
        {"name": "glm-5.2",               "type": "vision"},
        {"name": "glm-5.1",               "type": "vision"},
        {"name": "glm-5",                 "type": "vision"},
        {"name": "glm-4.7",               "type": "vision"},
        {"name": "kimi-k2.7-code",        "type": "vision"},
        {"name": "kimi-k2.6",             "type": "vision"},
        {"name": "kimi-k2.5",             "type": "vision"},
        {"name": "kimi-k2-thinking",      "type": "vision"},
        {"name": "MiniMax-M3",            "type": "vision"},
        {"name": "MiniMax-M2.7",          "type": "vision"},
        {"name": "MiniMax-M2.5",          "type": "vision"},
        {"name": "MiniMax-M2.1",          "type": "vision"},
        {"name": "codeqwen1.5-7b-chat",   "type": "text"},
    ],
    "DEEPSEEK": [
        {"name": "deepseek-v4-pro",       "type": "text"},
        {"name": "deepseek-v4-flash",     "type": "text"},
    ],
}

# Runtime model catalog — starts as built-in, can be replaced by
# --models-file load or --fetch.
_MODELS: Dict[str, List[Dict[str, str]]] = _BUILTIN_MODELS


def _load_models_file(path: str) -> Dict[str, List[Dict[str, str]]]:
    """Load model catalog from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data


def _save_models_file(path: str, models: Dict[str, List[Dict[str, str]]]):
    """Save model catalog to a JSON file."""
    with open(path, 'w') as f:
        json.dump(models, f, indent=2, ensure_ascii=False)


def _fetch_models_from_api() -> Dict[str, List[Dict[str, str]]]:
    """Query /v1/models from each configured provider, return a catalog."""
    result: Dict[str, List[Dict[str, str]]] = {}
    for prov in ("QWEN", "DEEPSEEK"):
        key = os.environ.get(f"{prov}_API_KEY", "")
        url = os.environ.get(f"{prov}_BASE_URL", "")
        if not key or not url:
            print(f"  {prov}: (no credentials — skipped)")
            continue
        try:
            client = OpenAI(api_key=key, base_url=url,
                            timeout=httpx.Timeout(10.0, connect=5.0))
            resp = client.models.list()
            names = sorted(m.id for m in resp.data)
            result[prov] = [
                {"name": n, "type": "vision" if _is_vision_model(n) else "text"}
                for n in names
            ]
            print(f"  {prov}: {len(names)} models from {url}")
        except Exception as e:
            print(f"  {prov}: ✗ {type(e).__name__}: {e}")
    return result


def _resolve_models(args) -> Dict[str, List[Dict[str, str]]]:
    """Resolve model catalog based on CLI args.  Mutates global _MODELS."""
    global _MODELS

    if args.fetch:
        # Fetch from live API
        print("\n── Fetching from /v1/models ──")
        models = _fetch_models_from_api()
        if args.models_file:
            _save_models_file(args.models_file, models)
            total = sum(len(v) for v in models.values())
            print(f"Saved {total} models → {args.models_file}")
        _MODELS = models
        return models
    elif args.models_file:
        # Load from user-specified file
        try:
            models = _load_models_file(args.models_file)
            _MODELS = models
            return models
        except FileNotFoundError:
            print(f"ERROR: {args.models_file} not found.")
            print(f"  Run with --fetch first to create it:")
            print(f"  python test_api.py --list-models --fetch --models-file {args.models_file}")
            sys.exit(1)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"ERROR: {args.models_file}: {e}")
            sys.exit(1)
    else:
        # Use built-in
        _MODELS = _BUILTIN_MODELS
        return _BUILTIN_MODELS


def print_model_list(models: Dict[str, List[Dict[str, str]]]):
    """Print models grouped by provider."""
    for prov in ("QWEN", "DEEPSEEK"):
        print(f"\n── {prov} ──")
        ml = models.get(prov, [])
        if not ml:
            print("  (no models listed)")
            continue
        for m in ml:
            print(f"  {m['name']:<40s} ({m['type']})")
    print()

# Model prefix → (provider_key).  First match wins.
# Provider key maps to {KEY}_API_KEY / {KEY}_BASE_URL env vars.
# For third-party models served through the QWEN MAAS gateway
# (glm, kimi, minimax, etc.), use --provider QWEN explicitly
# because the model name itself doesn't indicate the route.
_MODEL_PROVIDER_MAP = [
    ("qwq",       "QWEN"),
    ("qvq",       "QWEN"),
    ("qwen",      "QWEN"),
    ("codeqwen",  "QWEN"),
    ("deepseek",  "DEEPSEEK"),
    ("gpt",       "OPENAI"),
    ("gemini",    "GEMINI"),
    ("claude",    "ANTHROPIC"),
]

# Third-party prefixes that the QWEN MAAS gateway proxies.
# These don't have their own API keys — access them via --provider QWEN.
_MAAS_THIRD_PARTY_PREFIXES = (
    "glm", "kimi", "minimax", "gui", "siliconflow",
    "zhipu", "fun-asr", "wan2", "z-image", "xiaomi",
    "vanchin", "tongyi", "sre-", "test-",
)


def _classify(model: str) -> Tuple[Optional[str], bool, bool]:
    """Return (provider_key, is_qwen, is_deepseek) for a model name.

    is_qwen / is_deepseek control API parameter selection
    (e.g. enable_thinking for Qwen, thinking.type for DeepSeek).
    """
    ml = model.lower()

    # ── Qwen-native & Qwen-reasoning models ────────────────────────────
    for prefix in ("qwq", "qvq", "qwen", "codeqwen"):
        if ml.startswith(prefix):
            return "QWEN", True, False

    # ── DeepSeek-native models ─────────────────────────────────────────
    if ml.startswith("deepseek"):
        return "DEEPSEEK", False, True

    # ── Other known providers (OpenAI, Gemini, Claude, etc.) ───────────
    for prefix, prov in _MODEL_PROVIDER_MAP:
        if ml.startswith(prefix):
            return prov, False, False

    # ── MAAS third-party (glm, kimi, minimax, …) ───────────────────────
    # These are accessed through the QWEN MAAS gateway and behave like
    # generic OpenAI-compatible models (no Qwen-specific params).
    for prefix in _MAAS_THIRD_PARTY_PREFIXES:
        if ml.startswith(prefix):
            return "QWEN", False, False

    # ── Fallback ───────────────────────────────────────────────────────
    # Try to guess from _MODELS (built-in or loaded from file).
    ml_lower = ml.lower()
    for prov, models in _MODELS.items():
        for m in models:
            if m["name"].lower() == ml_lower:
                return prov, False, False

    # Last resort
    return model.split("-")[0].upper(), False, False


def _resolve_provider(model: str, explicit: Optional[str] = None) -> str:
    """Return the env-var prefix (e.g. 'QWEN') for a model name."""
    if explicit:
        return explicit.upper()
    prov, _, _ = _classify(model)
    return prov


def _get_endpoint(model: str, provider: Optional[str] = None):
    """Resolve (api_key, base_url) from {PROVIDER}_API_KEY, {PROVIDER}_BASE_URL."""
    prov = _resolve_provider(model, provider)
    key = os.environ.get(f"{prov}_API_KEY", "")
    url = os.environ.get(f"{prov}_BASE_URL", "")
    return key, url


# ── model type detection ──────────────────────────────────────────────────

# MAAS routing prefixes — strip these before type detection so that
# "siliconflow/deepseek-v3.2" is recognized as a deepseek model, not
# a generic third-party model.
_MAAS_ROUTING_PREFIXES = (
    "siliconflow/", "vanchin/", "zhipu/",
)


def _is_vision_model(model: str) -> bool:
    """Auto-detect whether a model supports image input based on its name.

    Qwen (阿里云百炼 / MAAS):
      Vision:  vl, omni, ocr, qvq, qwq, plus, flash
               (e.g. qwen3.7-plus, qwen-vl-max, qwen3-omni-flash)
      Text:    max, turbo, long, coder, code, math-turbo, deep-research
               (e.g. qwen3.7-max, qwen-coder-turbo, codeqwen1.5-7b-chat)

    DeepSeek:
      Vision:  vl, ocr           (e.g. deepseek-vl, deepseek-ocr-2)
      Text:    everything else   (e.g. deepseek-v4-pro, deepseek-v3.2)
    """
    ml = model.lower()
    _, is_qwen, is_deepseek = _classify(model)

    # ── Strip MAAS routing prefixes ────────────────────────────────────
    # "siliconflow/deepseek-v3.2" → check "deepseek-v3.2"
    for prefix in _MAAS_ROUTING_PREFIXES:
        if ml.startswith(prefix):
            ml = ml[len(prefix):]
            _, is_qwen, is_deepseek = _classify(ml)
            break

    if is_qwen:
        # Explicit vision → always multimodal
        for kw in ("vl", "omni", "ocr", "qvq", "qwq"):
            if kw in ml:
                return True
        # Explicit text-only (checked *before* plus/flash to handle
        # edge cases like qwen-coder-plus, qwen-math-turbo).
        for kw in ("coder", "codeqwen", "max", "turbo", "long", "math-turbo",
                   "deep-research", "deep-search", "image"):
            if kw in ml:
                return False
        # plus / flash are vision-capable (but not coder-plus, handled above)
        if "plus" in ml or "flash" in ml:
            return True
        # Default: Qwen models are vision-capable unless marked text
        return True

    if is_deepseek:
        for kw in ("vl", "ocr"):
            if kw in ml:
                return True
        return False

    # ── Other providers / third-party via MAAS ──────────────────────────
    # Known text-only prefixes (OpenAI text models, etc.)
    for prefix in ("gpt-3.5", "gpt-4o-mini", "o1", "o3", "o4-mini"):
        if ml.startswith(prefix):
            return False
    # Default to vision for unknown models — most MAAS-proxied models
    # (GLM, Kimi, MiniMax) are multimodal.
    return True


def _clear_proxy_env():
    """Remove proxy env vars that httpx might choke on (e.g. socks://)."""
    backup = {}
    for k in ('ALL_PROXY', 'all_proxy'):
        if k in os.environ:
            backup[k] = os.environ.pop(k)
    return backup


def _restore_proxy_env(backup):
    os.environ.update(backup)


# ── network diagnostics ─────────────────────────────────────────────────────

def _diagnose_network(url: str) -> Dict[str, float]:
    """Measure DNS, TCP, SSL timings for a URL.  Pure socket-level, no HTTP body.

    Returns a dict with keys: dns, tcp, ssl, total (all in seconds).
    """
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    result: Dict[str, float] = {}

    # ── DNS ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        socket.getaddrinfo(host, port, family=socket.AF_UNSPEC,
                           type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        result["dns"] = time.perf_counter() - t0
        result["dns_error"] = str(e)
        result["tcp"] = 0
        result["ssl"] = 0
        result["total"] = result["dns"]
        return result
    result["dns"] = time.perf_counter() - t0

    # ── TCP ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect((host, port))
    except OSError as e:
        result["tcp"] = time.perf_counter() - t0
        result["tcp_error"] = str(e)
        sock.close()
        result["ssl"] = 0
        result["total"] = result["dns"] + result["tcp"]
        return result
    result["tcp"] = time.perf_counter() - t0

    # ── SSL (HTTPS only) ────────────────────────────────────────────────
    if parsed.scheme == "https":
        ctx = ssl.create_default_context()
        t0 = time.perf_counter()
        try:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                pass
        except (ssl.SSLError, OSError) as e:
            result["ssl"] = time.perf_counter() - t0
            result["ssl_error"] = str(e)
            result["total"] = result["dns"] + result["tcp"] + result["ssl"]
            return result
        result["ssl"] = time.perf_counter() - t0
    else:
        sock.close()
        result["ssl"] = 0

    result["total"] = result["dns"] + result["tcp"] + result["ssl"]
    return result


def _print_network_diag(diag: Dict[str, float]):
    """Pretty-print network diagnostic results."""
    print("── Network ──")
    for phase in ("dns", "tcp", "ssl"):
        label = phase.upper()
        val = diag.get(phase, 0)
        err = diag.get(f"{phase}_error", "")
        if err:
            print(f"  {label:<4s} ✗ {err}")
        else:
            ms = val * 1000
            slow = " ⚠ SLOW!" if ms > 200 else ""
            print(f"  {label:<4s} {ms:5.0f} ms{slow}")
    total_ms = diag.get("total", 0) * 1000
    print(f"  sum   {total_ms:5.0f} ms  (connection setup, no HTTP)")
    print()


# ── test helpers ──────────────────────────────────────────────────────────

def build_request(model: str) -> List[Dict]:
    """Build a minimal message list appropriate for the model type."""
    if _is_vision_model(model):
        return [{"role": "user", "content": [
            {"type": "text", "text": MINIMAL_TEXT},
            {"type": "image_url", "image_url": {"url": MINIMAL_IMAGE}},
        ]}]
    else:
        return [{"role": "user", "content": MINIMAL_TEXT}]


def _call_api(client: OpenAI, model: str, messages: List[Dict],
              is_qwen: bool, is_deepseek: bool,
              total_timeout: Optional[float] = None) -> dict:
    """Call the chat completions API with provider-appropriate parameters.

    Returns a dict with keys: ok, content, elapsed, error, response_obj

    Note: httpx's timeout is per-chunk idle time (resets between TCP reads),
    so an LLM server that streams tokens chunk-by-chunk may never trigger it.
    We use a daemon thread + join(timeout) to enforce a *total* wall-clock
    deadline.  Daemon threads don't block process exit when abandoned.
    """
    kwargs: Dict = dict(
        model=model,
        messages=messages,
        temperature=0.0,
    )

    # ── provider-specific parameters ────────────────────────────────────
    if is_qwen:
        # Qwen models use max_completion_tokens and support enable_thinking
        kwargs['max_completion_tokens'] = 32
        kwargs['extra_body'] = {'enable_thinking': False}
    elif is_deepseek:
        # DeepSeek: use max_tokens (not max_completion_tokens).
        # Don't send enable_thinking — it's a Qwen-specific param.
        kwargs['max_tokens'] = 32
        # DeepSeek V4 may return reasoning_content; we want the final answer.
        kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}
    else:
        kwargs['max_tokens'] = 32

    if total_timeout is None:
        # No total deadline — call directly with a simple spinner
        result: Dict = {'response': None, 'error': None}

        def _do_call():
            try:
                result['response'] = client.chat.completions.create(**kwargs)
            except Exception as exc:
                result['error'] = exc

        t = threading.Thread(target=_do_call, daemon=True)
        t0 = time.time()
        t.start()
        while t.is_alive():
            elapsed = time.time() - t0
            print(f"\r  ⏳ waiting... {elapsed:.0f}s", end="", flush=True)
            t.join(timeout=1.0)
        elapsed = time.time() - t0
        # Clear the progress line
        print("\r" + " " * 40 + "\r", end="", flush=True)
        if result['error'] is not None:
            raise result['error']
        response = result['response']
    else:
        # Enforce a total wall-clock deadline via daemon thread.
        # ThreadPoolExecutor.shutdown(wait=True) blocks on __exit__, so we
        # use a bare daemon thread with polled join(timeout=1.0) instead.
        result: Dict = {'response': None, 'error': None}

        def _do_call():
            try:
                result['response'] = client.chat.completions.create(**kwargs)
            except Exception as exc:
                result['error'] = exc

        t = threading.Thread(target=_do_call, daemon=True)
        t0 = time.time()
        t.start()
        while t.is_alive():
            remain = total_timeout - (time.time() - t0)
            if remain <= 0:
                break
            print(f"\r  ⏳ waiting... {time.time() - t0:.0f}s  (timeout {total_timeout:.0f}s)", end="", flush=True)
            t.join(timeout=1.0)

        if t.is_alive():
            # Deadline hit — don't wait for the thread.  It's a daemon so it
            # won't block process exit.
            elapsed = time.time() - t0
            print(f"\r  ✗ timeout after {elapsed:.1f}s" + " " * 20, flush=True)
            return {
                'ok': False,
                'content': '',
                'elapsed': elapsed,
                'error': f'Total timeout ({total_timeout:.0f}s) exceeded',
                'response_obj': None,
            }

        elapsed = time.time() - t0
        # Clear the progress line
        print("\r" + " " * 50 + "\r", end="", flush=True)
        if result['error'] is not None:
            raise result['error']
        response = result['response']

    msg = response.choices[0].message
    # Some reasoning models put the answer in reasoning_content, not content
    content = msg.content
    if not content:
        reasoning = getattr(msg, 'reasoning_content', None)
        if reasoning:
            content = reasoning

    return {
        'ok': bool(content),
        'content': content or '',
        'elapsed': elapsed,
        'error': None,
        'response_obj': response,
    }


def test_one(client: OpenAI, model: str, total_timeout: Optional[float] = None) -> bool:
    """Send one minimal request and report timing."""
    mode = "vision" if _is_vision_model(model) else "text-only"
    _, is_qwen, is_deepseek = _classify(model)
    print(f"\n── {model} ({mode}) ──")

    messages = build_request(model)

    try:
        result = _call_api(client, model, messages, is_qwen, is_deepseek,
                           total_timeout=total_timeout)
        elapsed = result['elapsed']
        ok = result['ok']
        status = "✓" if ok else "✗"
        slow = " ⚠ SLOW!" if elapsed > 10 else ""
        print(f"{status} {model}: {elapsed:.1f}s{slow}")
        if ok:
            print(f"  → {result['content'][:200]}")
        elif result['response_obj'] is not None:
            # Diagnostic: dump the full message object
            msg = result['response_obj'].choices[0].message
            print(f"  content: {msg.content!r}")
            rc = getattr(msg, 'reasoning_content', None)
            if rc:
                print(f"  reasoning_content: {rc[:200]!r}")
            print(f"  finish_reason: {result['response_obj'].choices[0].finish_reason!r}")
            if result['response_obj'].usage:
                u = result['response_obj'].usage
                print(f"  usage: prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")
        else:
            # Timeout or other error without a response object
            print(f"  error: {result['error']}")
        return ok
    except Exception as e:
        elapsed = time.time()
        print(f"✗ {model}: {type(e).__name__}: {e}")
        return False


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quick API latency test")
    parser.add_argument("--model", type=str,
                        help="Model name (e.g. qwen3.7-plus, deepseek-v4-pro)")
    parser.add_argument("--provider", type=str, default=None,
                        help="Provider name override (e.g. qwen, deepseek)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat N times (default: 1)")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Total wall-clock timeout in seconds (default: 15.0). "
                             "Unlike httpx's per-chunk idle timeout, this is a hard "
                             "deadline on the entire request.")
    parser.add_argument("--list-models", action="store_true",
                        help="Print model list and exit (from --models-file, "
                             "built-in, or --fetch).")
    parser.add_argument("--fetch", action="store_true",
                        help="Query /v1/models from configured providers. "
                             "Use with --models-file to save the result, or "
                             "with --list-models to see the live list.")
    parser.add_argument("--models-file", type=str, default=None,
                        help="Path to a JSON model-catalog file. "
                             "Create it: --fetch --models-file PATH. "
                             "Use it: --list-models --models-file PATH.")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run network diagnostic (DNS/TCP/SSL timing) "
                             "before the model test.")
    args = parser.parse_args()

    # ── fetch mode (may or may not include --list-models) ────────────────
    if args.fetch or args.models_file:
        _resolve_models(args)

    # ── list-models mode ─────────────────────────────────────────────────
    if args.list_models:
        print_model_list(_MODELS)
        sys.exit(0)

    # ── after --fetch without --list-models, just exit ───────────────────
    if args.fetch:
        sys.exit(0)

    # ── model is required for normal test mode ───────────────────────────
    if not args.model:
        parser.error("--model is required (or use --list-models to browse)")

    # ── resolve endpoint ─────────────────────────────────────────────────
    key, url = _get_endpoint(args.model, args.provider)
    if not key:
        prov = _resolve_provider(args.model, args.provider)
        print(f"ERROR: No API key found for provider '{prov}'.")
        print(f"  Expected env vars: {prov}_API_KEY, {prov}_BASE_URL")
        print(f"  Source .env.api or set them manually.")
        sys.exit(1)

    prov_key, is_qwen, is_deepseek = _classify(args.model)
    mode = "vision" if _is_vision_model(args.model) else "text-only"
    print(f"Model: {args.model} ({mode})  provider={prov_key}")
    print(f"URL:   {url}")
    print(f"Repeat: {args.repeat}")
    print(f"Timeout: {args.timeout:.0f}s (total wall-clock)")

    # ── network diagnostic (optional) ────────────────────────────────────
    if args.diagnose:
        diag = _diagnose_network(url)
        _print_network_diag(diag)

    # ── build client ─────────────────────────────────────────────────────
    # httpx timeout: per-chunk idle guard (may not fire on streaming servers).
    # Real enforcement is done via ThreadPoolExecutor total_timeout.
    proxy_backup = _clear_proxy_env()
    try:
        client = OpenAI(
            api_key=key,
            base_url=url,
            timeout=httpx.Timeout(args.timeout + 5.0, connect=10.0),
        )
    finally:
        _restore_proxy_env(proxy_backup)

    # ── run tests ────────────────────────────────────────────────────────
    ok_count = 0
    total = args.repeat
    for r in range(total):
        if total > 1:
            print(f"\n━━━ round {r + 1}/{total} ━━━")
        if test_one(client, args.model, total_timeout=args.timeout):
            ok_count += 1

    # ── summary ──────────────────────────────────────────────────────────
    if total > 1:
        print(f"\n{'─' * 40}")
        print(f"{args.model}: {ok_count}/{total} OK")


if __name__ == "__main__":
    main()
