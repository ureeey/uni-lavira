"""DashScope-native API client for LaViRA.

Uses the `dashscope` Python SDK (MultiModalConversation API) instead of the
OpenAI-compatible HTTP endpoint.  Message format and response parsing differ
from the OpenAI path; this module handles the conversion transparently so that
upstream callers (agent.py) are unchanged.
"""

from PIL import Image
from habitat import logger
import base64
import io
import json
import os
import sys
import time

import numpy as np

try:
    import dashscope
except ImportError:
    dashscope = None

# ── Logging verbosity (shared env vars with api_openai.py) ────────────
_LOG_PROMPT_LEVEL = int(os.environ.get("LAVIRA_LOG_PROMPT", "0"))
_LOG_VERBOSE = int(os.environ.get("LAVIRA_LOG_VERBOSE", "0"))
_LOG_NETWORK = int(os.environ.get("LAVIRA_LOG_NETWORK", "0"))
_LOG_BODY = int(os.environ.get("LAVIRA_LOG_BODY", "0"))

# ── HTTP-level body logging (intercept requests.Session.send) ────────
_orig_session_send = None


def _install_body_interceptor():
    """Monkey-patch requests.Session.send to log actual HTTP body size."""
    global _orig_session_send
    if _orig_session_send is not None:
        return  # already installed
    import requests
    _orig_session_send = requests.Session.send

    def _patched_send(self, request, **kwargs):
        body_len = len(request.body) if request.body else 0
        logger.info(f"[BODY HTTP] {request.method} {request.url}  "
                    f"body={body_len/1024:.0f} KB  "
                    f"content-type={request.headers.get('Content-Type', '?')}")
        return _orig_session_send(self, request, **kwargs)

    requests.Session.send = _patched_send


def _log_network(msg: str):
    if _LOG_NETWORK:
        logger.info(f"[NET] {msg}")


def _log_verbose(msg: str):
    if _LOG_VERBOSE == 0:
        logger.info(msg)


def _log_body(messages, label=""):
    """Log the actual message content structure — image format, count, sizes."""
    if not _LOG_BODY:
        return
    img_sources = []
    total_text_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_text_chars += len(content)
        elif isinstance(content, list):
            for item in content:
                if "text" in item:
                    total_text_chars += len(item["text"])
                elif "image" in item:
                    url = item["image"]
                    if url.startswith("data:"):
                        size_kb = len(url) / 1024
                        img_sources.append(f"base64 ({size_kb:.0f} KB)")
                    elif url.startswith("oss://"):
                        img_sources.append(f"oss://  ({len(url):.0f} B)")
                    elif url.startswith("http"):
                        img_sources.append(f"http   ({len(url):.0f} B)")
                    else:
                        img_sources.append(f"other  ({len(url):.0f} B)")

    total_body_kb = total_text_chars / 1024 + sum(
        len(item.get("image", "")) / 1024
        for msg in messages if isinstance(msg.get("content", []), list)
        for item in msg["content"] if "image" in item
    )

    summary = ", ".join(img_sources) if img_sources else "no images"
    logger.info(f"[BODY] {label}  {len(img_sources)} imgs  |  {summary}  |  "
                f"text: {total_text_chars/1024:.0f} KB  body: {total_body_kb:.0f} KB")


class _LatencyEstimator:
    """Online estimation of API latency model:  latency = x + output_tokens / k.

    Uses decoupled exponential moving average updates after each call.
    """

    def __init__(self, x0=3.0, k0=30.0, alpha=0.3):
        self.x = x0       # fixed overhead (seconds)
        self.k = k0       # token generation speed (tokens / second)
        self.alpha = alpha
        self.x_max = 0.0          # peak overhead (excludes initial x0)
        self.k_min = float('inf') # slowest speed (excludes initial k0)

    def update(self, output_tokens: int, elapsed_sec: float):
        """Incorporate a new observation (n output tokens, t seconds)."""
        n = max(output_tokens, 1)
        t = max(elapsed_sec, 0.1)

        # ── Update x (overhead) ──────────────────────────────────────
        x_obs = t - n / max(self.k, 1)
        x_obs = max(0.1, min(60.0, x_obs))
        self.x = self.alpha * x_obs + (1 - self.alpha) * self.x
        if self.x > self.x_max:
            self.x_max = self.x

        # ── Update k (token speed) ───────────────────────────────────
        # Only update k when the model is physically consistent (x < t).
        # When x >= t, the overhead estimate exceeds the total time,
        # making k_obs = n/0.1 meaningless — skip to avoid a vicious
        # cycle where high x → high k → small n/k → x stays high.
        net = t - self.x
        if net > 0.1:
            k_obs = n / net
            k_obs = max(5.0, min(500.0, k_obs))
            self.k = self.alpha * k_obs + (1 - self.alpha) * self.k
            if self.k < self.k_min:
                self.k_min = self.k


class LaViRA_DashScope_API:
    """DashScope-native multimodal API client.

    Parameters
    ----------
    dashscope_api_key : str
        DashScope API key (sk-…).
    la_model_name : str
        Model name for Language Agent calls.
    va_model_name : str or None
        Model name for Vision Agent calls (defaults to *la_model_name*).
    """

    def __init__(self, dashscope_api_key, la_model_name, va_model_name=None,
                 dashscope_base_url=None):
        if dashscope is None:
            raise ImportError(
                "dashscope package is required for DashScope mode. "
                "Install it with: pip install dashscope"
            )
        self.dashscope_api_key = dashscope_api_key
        self.la_model_name = la_model_name
        self.va_model_name = va_model_name or la_model_name

        # Resolve DashScope base URL:
        #   - If dashscope_base_url is explicitly provided (--dashscope-maas flag),
        #     use the MaaS workspace /api/v1 for dedicated resources.
        #   - Otherwise, default to the public dashscope.aliyuncs.com (no body limit).
        if dashscope_base_url:
            self._base_url = dashscope_base_url
        else:
            self._base_url = 'https://dashscope.aliyuncs.com/api/v1'

        # Point the SDK at our chosen endpoint before making any calls
        dashscope.base_http_api_url = self._base_url

        logger.info(f"[DashScope native] LA={self.la_model_name}  VA={self.va_model_name}")
        logger.info(f"[DashScope native] base_url={self._base_url}")

        if _LOG_BODY:
            _install_body_interceptor()

        self.reset_stats()
        self._la_round = 0
        self._va_round = 0
        self._lat_la = _LatencyEstimator(x0=3.0, k0=30.0, alpha=0.3)
        self._lat_va = _LatencyEstimator(x0=3.0, k0=30.0, alpha=0.3)

    # ── stats ─────────────────────────────────────────────────────────

    def reset_stats(self):
        self.stats = {
            'Language Action Model': {
                'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
                'elapsed_total': 0.0, 'elapsed_max': 0.0, 'elapsed_min': float('inf'),
            },
            'Vision Action Model': {
                'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
                'elapsed_total': 0.0, 'elapsed_max': 0.0, 'elapsed_min': float('inf'),
            },
        }

    def get_usage_stats(self):
        return self.stats.copy()

    def get_model_info(self):
        return {
            "primary_model": self.la_model_name,
            "secondary_model": self.va_model_name,
            "has_secondary": self.va_model_name != self.la_model_name,
            "use_dashscope": True,
        }

    def print_usage_stats(self):
        la = self.stats['Language Action Model']
        va = self.stats['Vision Action Model']

        def _fmt_time(s):
            if s == 0 or s == float('inf'):
                return '    -'
            return f'{s:5.1f}s'

        def _print_section(title, model_name, s, lat):
            n = s['calls']
            if n == 0:
                return
            logger.info(f"  {title} ({model_name}):")
            logger.info(f"    Calls:          {n:>6d}")
            logger.info(f"    Time total:     {_fmt_time(s['elapsed_total'])}")
            logger.info(f"    Time avg:       {_fmt_time(s['elapsed_total'] / n)}")
            logger.info(f"    Time max:       {_fmt_time(s['elapsed_max'])}")
            logger.info(f"    Time min:       {_fmt_time(s['elapsed_min'] if s['elapsed_min'] != float('inf') else 0)}")
            logger.info(f"    Input tokens:   {s['input_tokens']:>8,}")
            logger.info(f"    Output tokens:  {s['output_tokens']:>8,}")
            logger.info(f"    Total tokens:   {s['total_tokens']:>8,}")
            logger.info(f"    Avg out / call: {s['output_tokens'] / n:>8.0f}")
            logger.info(f"    Max latency x:  {_fmt_time(lat.x_max)}")
            logger.info(f"    Min latency k:  {lat.k_min if lat.k_min != float('inf') else 0:>5.0f} tok/s")

        logger.info("==================== MODEL USAGE STATISTICS ====================")
        _print_section("LA", self.la_model_name, la, self._lat_la)
        _print_section("VA", self.va_model_name, va, self._lat_va)

        # ── Combined ─────────────────────────────────────────────────
        total_calls = la['calls'] + va['calls']
        total_tokens = la['total_tokens'] + va['total_tokens']
        total_elapsed = la['elapsed_total'] + va['elapsed_total']
        total_out = la['output_tokens'] + va['output_tokens']
        comb_max = max(la['elapsed_max'], va['elapsed_max'])
        comb_min_min = min(la['elapsed_min'], va['elapsed_min'])
        comb_min = comb_min_min if comb_min_min != float('inf') else 0

        logger.info(f"  ───────────────────────────────────────────")
        logger.info(f"  COMBINED:")
        logger.info(f"    Calls:          {total_calls:>6d}")
        logger.info(f"    Time total:     {_fmt_time(total_elapsed)}")
        logger.info(f"    Time avg:       {_fmt_time(total_elapsed / max(1, total_calls))}")
        logger.info(f"    Time max:       {_fmt_time(comb_max)}")
        logger.info(f"    Time min:       {_fmt_time(comb_min)}")
        logger.info(f"    Input tokens:   {la['input_tokens'] + va['input_tokens']:>8,}")
        logger.info(f"    Output tokens:  {total_out:>8,}")
        logger.info(f"    Total tokens:   {total_tokens:>8,}")
        logger.info(f"    Avg out / call: {total_out / max(1, total_calls):>8.0f}")
        logger.info("================================================================")
        return {
            'la': la.copy(), 'va': va.copy(),
            'total_calls': total_calls, 'total_tokens': total_tokens,
        }

    def eval(self):
        """Compatibility no-op."""

    # ── debug ──────────────────────────────────────────────────────────

    def _save_debug_info(self, log_path, messages, response_text):
        if not log_path:
            return
        try:
            os.makedirs(log_path, exist_ok=True)
            saved_messages = []
            img_count = 0
            for msg in messages:
                new_msg = {'role': msg['role'], 'content': []}
                if isinstance(msg['content'], list):
                    for item in msg['content']:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            url = item['image_url']['url']
                            if url.startswith('data:image/'):
                                try:
                                    header, encoded = url.split(',', 1)
                                    data = base64.b64decode(encoded)
                                    img_filename = f"image_{img_count}.png"
                                    img_path = os.path.join(log_path, img_filename)
                                    with open(img_path, 'wb') as f:
                                        f.write(data)
                                    new_msg['content'].append({
                                        'type': 'image_url',
                                        'image_url': {'url': img_filename},
                                    })
                                    img_count += 1
                                except Exception as e:
                                    logger.error(f"Failed to save image: {e}")
                                    new_msg['content'].append({'type': 'image_url', 'image_url': {'url': 'FAILED_TO_SAVE'}})
                            else:
                                new_msg['content'].append(item)
                        else:
                            new_msg['content'].append(item)
                else:
                    new_msg['content'] = msg['content']
                saved_messages.append(new_msg)
            with open(os.path.join(log_path, 'prompt.json'), 'w') as f:
                json.dump(saved_messages, f, indent=2)
            with open(os.path.join(log_path, 'response.txt'), 'w') as f:
                f.write(str(response_text))
        except Exception as e:
            logger.error(f"Failed to save debug info: {e}")

    # ── message conversion ─────────────────────────────────────────────

    @staticmethod
    def _to_dashscope_messages(messages):
        """Convert OpenAI-format messages to DashScope MultiModalConversation format.

        OpenAI:   ``{"type": "text", "text": "..."}`` / ``{"type": "image_url", "image_url": {"url": "..."}}``
        DashScope: ``{"text": "..."}`` / ``{"image": "..."}``
        """
        converted = []
        for msg in messages:
            new_msg = {"role": msg["role"]}
            content = msg.get("content", "")
            if isinstance(content, str):
                new_msg["content"] = content
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "text":
                        new_content.append({"text": item["text"]})
                    elif item.get("type") == "image_url":
                        new_content.append({"image": item["image_url"]["url"]})
                new_msg["content"] = new_content
            converted.append(new_msg)
        return converted

    # ── generation ─────────────────────────────────────────────────────

    def generate(self, messages, images=None, max_new_tokens=1024, temperature=0.7,
                 use_la=False, log_path=None, retries=0, max_retries=5, **kwargs):
        """Unified generate interface — identical signature to the OpenAI path."""
        return self._generate_dashscope(
            messages=messages, max_new_tokens=max_new_tokens, temperature=temperature,
            use_la=use_la, log_path=log_path, retries=retries, max_retries=max_retries,
            **kwargs,
        )

    def _generate_dashscope(self, messages, max_new_tokens=1024, temperature=0.7,
                            use_la=False, log_path=None, retries=0, max_retries=5, **kwargs):
        model_name = self.la_model_name if (use_la and self.la_model_name) else self.va_model_name
        stats_key = 'Language Action Model' if use_la else 'Vision Action Model'

        if use_la:
            self._la_round += 1
            label = f"LA #{self._la_round}"
        else:
            self._va_round += 1
            label = f"VA #{self._va_round}"

        _msg_str = json.dumps(messages, ensure_ascii=False)
        _payload_kb = len(_msg_str.encode('utf-8')) / 1024
        if _LOG_VERBOSE:
            sys.stdout.write("\n")
            sys.stdout.flush()

        _log_network(f"{label}  [DashScope] target={model_name}  payload={_payload_kb:.0f}KB")
        logger.info(f"▐ {label} → {model_name}  {_payload_kb:.0f} KB  [DashScope]")

        try:
            ds_messages = self._to_dashscope_messages(messages)
            _log_body(ds_messages, f"{label} DashScope msg format")

            # Extract DashScope-specific parameters from extra_body.
            # Default enable_thinking=False to match OpenAI-path behaviour;
            # the DashScope default (True) causes verbose thinking traces that
            # inflate output tokens ~9× and degrade structured-output quality.
            extra_body = kwargs.pop('extra_body', None) or {}
            ds_kwargs = {
                'enable_thinking': extra_body.get('enable_thinking', False),
            }

            _call_t = time.time()
            response = dashscope.MultiModalConversation.call(
                api_key=self.dashscope_api_key,
                model=model_name,
                messages=ds_messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                **ds_kwargs,
            )
            _elapsed = time.time() - _call_t

            if response.status_code != 200:
                err_msg = f"DashScope API error: code={response.code} message={response.message}"
                logger.error(f"[{stats_key}] {err_msg}")
                raise Exception(err_msg)

            # Update usage statistics
            self.stats[stats_key]['calls'] += 1
            if response.usage:
                input_tokens = getattr(response.usage, 'input_tokens', 0) or 0
                output_tokens = getattr(response.usage, 'output_tokens', 0) or 0
                total_tokens = getattr(response.usage, 'total_tokens', 0) or 0
                self.stats[stats_key]['input_tokens'] += input_tokens
                self.stats[stats_key]['output_tokens'] += output_tokens
                self.stats[stats_key]['total_tokens'] += total_tokens
                # Track elapsed time
                s = self.stats[stats_key]
                s['elapsed_total'] += _elapsed
                if _elapsed > s['elapsed_max']:
                    s['elapsed_max'] = _elapsed
                if _elapsed < s['elapsed_min']:
                    s['elapsed_min'] = _elapsed
                # Update latency model:  t = x + out/k
                _lat = self._lat_la if use_la else self._lat_va
                _lat.update(output_tokens, _elapsed)
                logger.info(f"▐ {label}  ✓ {_elapsed:.1f}s  |  "
                            f"in:{input_tokens}  out:{output_tokens}  total:{total_tokens}  "
                            f"x={_lat.x:.1f}s  k={_lat.k:.0f}tok/s")
                _slow = "  ⚠ SLOW!" if _elapsed > 30 else ""
                _log_network(f"{label}  ✓ {_elapsed:.1f}s  "
                             f"in={input_tokens}  out={output_tokens}{_slow}")

            # Parse response content — DashScope returns a list of part dicts
            output = response.output
            if output and output.choices:
                content_parts = output.choices[0].message.content
                if isinstance(content_parts, list):
                    text = ''.join(
                        part.get('text', '') for part in content_parts if 'text' in part
                    )
                else:
                    text = str(content_parts) if content_parts else ""
                self._save_debug_info(log_path, messages, text)
                return text

            logger.error(f"[{stats_key}] DashScope returned no choices")
            return "Error: No response content from DashScope"

        except Exception as e:
            err_str = str(e)
            _elapsed = time.time()
            logger.error(f"[{stats_key}] DashScope error  "
                         f"model={model_name} | {type(e).__name__}: {e}")
            _log_network(f"{label}  ✗ FAIL  {type(e).__name__}: {str(e)[:200]}")

            non_recoverable = (
                'data_inspection_failed' in err_str or
                'DataInspectionFailed' in err_str or
                'inappropriate content' in err_str or
                'invalid_request_error' in err_str or
                'InvalidParameter' in err_str
            )
            if non_recoverable:
                logger.error("Non-recoverable error; skipping retries.")
                return "Error: API rejected request (non-recoverable)"
            if retries >= max_retries:
                logger.error(f"Max retries ({max_retries}) reached. Giving up.")
                return "Error: Failed to get response from DashScope after max retries"

            logger.info(f'Forcing retry ({retries + 1}/{max_retries})..')
            time.sleep(30)
            return self._generate_dashscope(
                messages=messages, max_new_tokens=max_new_tokens, temperature=temperature,
                use_la=use_la, log_path=log_path, retries=retries + 1,
                max_retries=max_retries, extra_body=extra_body,
            )
